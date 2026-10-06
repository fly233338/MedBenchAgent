"""Supervisor-facing preparation entrypoint.

The business state machine lives in :mod:`medbenchagent.preparation.graph`.
This module freezes source inputs, invokes the two-stage graph, and exposes the
small LangGraph wrapper consumed by the Supervisor workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source, verify_source
from medbenchagent.preparation.graph import run_preparation_graph
from medbenchagent.preparation.worker import OpenHandsWorker
from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.dataset_runtime import dataset_runtime
from medbenchagent.runtime.run_artifacts import atomic_write_json


def _workspace(context: dict[str, Any]) -> Path:
    dataset_id = str(context.get("dataset_scope") or context.get("dataset_id") or "dataset")
    return Path(context["run_directory"]) / "preparation" / dataset_id


def _source_setup(context: dict[str, Any], workspace: Path) -> tuple[list[dict], dict[str, str]]:
    sources = [Source.model_validate(item).model_dump(mode="json") for item in context.get("sources", [])]
    if not sources:
        raise ValueError("Dataset preparation requires at least one source")
    bundle_root = workspace / "inputs"
    manifests = dict(context.get("source_manifests") or {})
    for source in sources:
        manifest = Path(manifests.get(source["id"]) or bundle_root / "source_manifests" / source["id"])
        inventory_source(Source.model_validate(source), manifest)
        verify_source(manifest)
        manifests[source["id"]] = str(manifest)
    return sources, manifests


def _load_checkpoint(workspace: Path) -> dict[str, Any]:
    path = workspace / "checkpoint.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _persist_checkpoint(workspace: Path, state: dict[str, Any]) -> None:
    # This is a diagnostic projection. LangGraph's checkpoint remains the
    # scheduler authority when the preparation graph is compiled by a caller.
    def normalize(value):
        if hasattr(value, "model_dump"):
            return normalize(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    safe = normalize({key: value for key, value in state.items() if key not in {"sources", "user_decisions"}})
    atomic_write_json(workspace / "checkpoint.json", safe)


def run_preparation(context: dict[str, Any]) -> dict[str, Any]:
    """Run the two coding stages and return a candidate handoff."""
    context = dict(context)
    workspace = _workspace(context)
    workspace.mkdir(parents=True, exist_ok=True)
    sources, manifests = _source_setup(context, workspace)
    # checkpoint.json is only a projection.  Reuse references after validating
    # their content; never copy its stage/status/accepted flags into routing.
    projection = _load_checkpoint(workspace)
    checkpoint = {
        key: projection[key]
        for key in ("accepted_handoff_ref", "resume_state_ref", "attempt_id", "session_seq", "attempt_mode")
        if key in projection
    }
    state = {
        **checkpoint,
        **context,
        "sources": sources,
        "source_roots": {item["id"]: item["root"] for item in sources},
        "source_manifests": manifests,
        "dataset_id": str(context.get("dataset_scope") or context.get("dataset_id") or "dataset"),
        "preparation_root": str(workspace),
        "user_request": context.get("user_request") or context.get("user_requirement") or "",
        "strict_publication": True,
        "canonical_validation": True,
    }
    state["notes"] = str(context.get("notes") or "")
    state["target_issue_ids"] = list(context.get("issue_ids") or [])
    runtime = dataset_runtime(context)
    outcome = run_preparation_graph(state, worker=OpenHandsWorker(runtime=runtime), max_attempts=int(context.get("preparation_max_attempts", 3)))
    _persist_checkpoint(workspace, outcome)
    status = outcome.get("status", "blocked")
    stage_result = outcome.get("stage_result") or {}
    updates = {
        "preparation_workspace": str(workspace),
        "preparation_status": "complete" if status == "complete" else status,
        "preparation_conversation_id": stage_result.get("conversation_id", ""),
        "preparation_termination_reason": stage_result.get("termination_reason", ""),
    }
    if outcome.get("preparation_candidate"):
        updates["preparation_candidate"] = outcome["preparation_candidate"]
    accepted = outcome.get("accepted_artifacts") or {}
    population = accepted.get("image_population.jsonl") if isinstance(accepted, dict) else None
    mapping = accepted.get("mapping.json") if isinstance(accepted, dict) else None
    if isinstance(population, dict):
        updates["preparation_population_path"] = population.get("path")
    if isinstance(mapping, dict):
        updates["preparation_mapping_digest"] = mapping.get("digest")
    if outcome.get("pending_questions"):
        updates["preparation_question"] = outcome["pending_questions"]
    if outcome.get("diagnostics"):
        updates["preparation_validation_issues"] = json.dumps(outcome["diagnostics"], ensure_ascii=False)
    if status == "complete":
        return {"status": "complete", "summary": "Preparation candidate submitted for host validation", "updates": updates}
    if status == "needs_input":
        return {"status": "needs_input", "summary": "Preparation is waiting for a source or semantic decision", "updates": updates}
    return {"status": "blocked" if status == "blocked" else "failed", "summary": outcome.get("diagnostic") or "Preparation stage failed", "updates": updates}


def mark_preparation_complete(workspace: str | Path) -> None:
    path = Path(workspace) / "checkpoint.json"
    state = _load_checkpoint(Path(workspace))
    state.update({"status": "complete", "stage": "publish"})
    atomic_write_json(path, state)


def mark_preparation_blocked(workspace: str | Path, reason: str = "validation_failed") -> None:
    path = Path(workspace) / "checkpoint.json"
    state = _load_checkpoint(Path(workspace))
    state.update({"status": "blocked", "termination_reason": reason})
    atomic_write_json(path, state)


class PreparationAgentState(TypedDict, total=False):
    context: dict[str, Any]
    result: dict[str, Any]


def _preparation_event(context: dict[str, Any], event_type: str, **values: object) -> None:
    path = context.get("construction_events_path")
    if not path:
        return
    payload = {
        "run_id": context.get("run_id"),
        "agent": "DatasetProcessingAgent",
        "dataset_id": context.get("dataset_scope"),
        "conversation_id": context.get("preparation_conversation_id"),
        "workspace": context.get("preparation_workspace"),
    }
    payload.update(values)
    record_event(path, event_type, **payload)


def run_preparation_agent(context: dict[str, Any]) -> dict[str, Any]:
    context = dict(context)
    _preparation_event(context, "agent_started", operation="process_dataset", status="started",
                       source_count=len(context.get("sources", [])))
    try:
        outcome = run_preparation(context)
    except ValueError as exc:
        outcome = {"status": "blocked", "summary": str(exc), "updates": {}}
    updates = outcome.get("updates", {})
    _preparation_event({**context, **updates}, "worker_result", operation="preparation_graph",
                       status=outcome["status"], diagnostic=outcome.get("summary", ""),
                       termination_reason=updates.get("preparation_termination_reason"))
    _preparation_event({**context, **updates}, "agent_finished", operation="process_dataset",
                       status=outcome["status"], diagnostic=outcome.get("summary", ""),
                       candidate_path=(updates.get("preparation_candidate") or {}).get("candidate_path"))
    return outcome


def compile_preparation_agent(*, checkpointer=None):
    def execute(state: PreparationAgentState):
        context = dict(state["context"])
        outcome = run_preparation_agent(context)
        context.update(outcome.get("updates", {}))
        context["preparation_agent_result"] = {key: value for key, value in outcome.items() if key != "updates"}
        return {"context": context, "result": context["preparation_agent_result"]}

    graph = StateGraph(PreparationAgentState)
    graph.add_node("preparation_agent", execute)
    graph.add_edge(START, "preparation_agent")
    graph.add_edge("preparation_agent", END)
    return graph.compile(checkpointer=checkpointer)
