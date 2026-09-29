"""The long-lived OpenHands dataset processing agent.

This module is the preparation entrypoint.  OpenHands owns exploration, mapping,
code, execution and repair; LangGraph only persists the session boundary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medbenchagent.runtime.dataset_runtime import dataset_runtime
from medbenchagent.runtime.preparation_task import build_data_preparation_task
from medbenchagent.runtime.preparation_session import CodingSession, load_session
from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source
from medbenchagent.data.sources import verify_source
from medbenchagent.runtime.construction import record_event


class DatasetAgentState(TypedDict, total=False):
    context: dict
    result: dict


DATASET_PROCESSING_AGENT = "DatasetProcessingAgent"


def _dataset_event(context: dict, event_type: str, **values: object) -> None:
    path = context.get("construction_events_path")
    if not path:
        return
    payload = {"run_id": context.get("run_id"), "agent": DATASET_PROCESSING_AGENT,
               "dataset_id": context.get("dataset_scope"),
               "conversation_id": context.get("preparation_conversation_id"),
               "workspace": context.get("preparation_workspace")}
    payload.update(values)
    record_event(path, event_type, **payload)


def _task(context: dict, workspace: Path):
    return build_data_preparation_task(context, workspace, feedback=context.get("feedback", ""))


def run_dataset_agent(context: dict):
    context = dict(context)
    _dataset_event(context, "agent_started", operation="prepare_data", status="started",
                   source_count=len(context.get("sources", [])))
    if not context.get("active_source"):
        sources = [Source.model_validate(item) for item in context.get("sources", [])]
        if not sources:
            return {"status": "blocked", "summary": "Dataset has no configured source", "updates": {}}
        source = sources[0]
        # Freeze manifests for every configured source up front.  The coding
        # worker still receives one immutable mount for compatibility, while
        # the task context records the complete source set and its manifests.
        manifest = Path(context["bundle_root"]) / "source_manifests" / source.id
        for configured in sources:
            inventory_source(configured, Path(context["bundle_root"]) / "source_manifests" / configured.id)
            _dataset_event(context, "source_manifest_created", operation="inventory",
                           source_id=configured.id,
                           source_manifest=str(Path(context["bundle_root"]) / "source_manifests" / configured.id))
        context.update(source_scope=source.id, active_source=source.model_dump(),
                       source_root=str(Path(source.root).resolve()), source_manifest=str(manifest),
                       source_manifests={item.id: str(Path(context["bundle_root"]) / "source_manifests" / item.id)
                                        for item in sources})
    source = context["active_source"]
    # All sources share one workspace and one conversation. The host submits
    # one complete task instead of scheduling inventory/mapping/repair stages.
    workspace = Path(context["run_directory"]) / "workers" / "openhands" / context["dataset_scope"]
    workspace.mkdir(parents=True, exist_ok=True)
    task = _task(context, workspace)
    _dataset_event(context, "session_started", operation="coding_session",
                   conversation_id=task.conversation_id, workspace=str(workspace),
                   source_ids=[item.get("id") for item in context.get("sources", [])])
    for manifest in context.get("source_manifests", {}).values():
        verify_source(manifest)
    runtime = dataset_runtime(context)
    _dataset_event(context, "worker_started", operation="coding_worker", backend=(context.get("execution_runtime") or {}).get("backend"),
                   conversation_id=task.conversation_id, required_outputs=task.required_outputs)
    def run_turn(payload):
        result = runtime.run(task.model_copy(update={"conversation_id": payload["conversation_id"],
                                                     "objective": payload["message"]})).model_dump()
        result["status"] = {"complete": "submitted", "failed": "blocked"}.get(result["status"], result["status"])
        return result
    session = CodingSession(workspace, task.conversation_id, run_turn)
    previous = load_session(workspace)
    coding = asyncio.run(session.continue_with(task.objective) if previous.get("conversation_id")
                         else session.run(task.objective))
    _dataset_event(context, "worker_result", operation="coding_worker", status=coding.status,
                   conversation_id=coding.conversation_id, candidate=coding.candidate,
                   diagnostic=coding.diagnostic, question=coding.question)
    if coding.status == "submitted":
        candidate_root = workspace / "outputs"
        payload = {"candidate_path": str(candidate_root), "source_ids": [item["id"] for item in context.get("sources", [])],
                   "status": "submitted"}
        updates = {"preparation_conversation_id": coding.conversation_id,
                   "preparation_workspace": str(workspace),
                   "preparation_status": "complete", "preparation_candidate": payload}
        _dataset_event({**context, **updates}, "candidate_submitted", operation="candidate",
                       status="submitted", candidate_path=str(candidate_root), source_ids=payload["source_ids"])
        _dataset_event({**context, **updates}, "agent_finished", operation="prepare_data", status="submitted",
                       candidate_path=str(candidate_root), source_ids=payload["source_ids"])
        return {"status": "complete", "summary": "Dataset processing agent completed candidate preparation",
                "updates": updates}
    updates = {"preparation_conversation_id": coding.conversation_id,
               "preparation_workspace": str(workspace),
               "preparation_status": coding.status}
    business_status = coding.status if coding.status in {"needs_input", "blocked"} else "blocked"
    _dataset_event({**context, **updates}, "agent_finished", operation="prepare_data",
                   status=business_status, diagnostic=coding.diagnostic, question=coding.question)
    return {"status": business_status,
            "summary": coding.diagnostic or coding.question,
            "updates": updates}


def compile_dataset_agent(*, checkpointer=None):
    def execute(state: DatasetAgentState):
        context = dict(state["context"])
        outcome = run_dataset_agent(context)
        context.update(outcome.get("updates", {}))
        context["dataset_agent_result"] = {k: v for k, v in outcome.items() if k != "updates"}
        return {"context": context, "result": context["dataset_agent_result"]}

    graph = StateGraph(DatasetAgentState)
    graph.add_node("dataset_processing_agent", execute)
    graph.add_edge(START, "dataset_processing_agent")
    graph.add_edge("dataset_processing_agent", END)
    return graph.compile(checkpointer=checkpointer)



