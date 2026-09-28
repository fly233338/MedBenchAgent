"""The long-lived OpenHands dataset processing agent.

This module is the preparation entrypoint.  OpenHands owns exploration, mapping,
code, execution and repair; LangGraph only persists the session boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medbenchagent.runtime.dataset_runtime import DatasetAgentTask, dataset_runtime
from medbenchagent.runtime.preparation_session import load_session, write_session
from medbenchagent.settings import get_settings
from medbenchagent.data.contracts import Source
from medbenchagent.data.sources import inventory_source


DATASET_WORKFLOW_PROMPT = """You are the dataset processing coding agent. Work continuously in one OpenHands
conversation until preparation is complete. Explore the complete read-only source, inspect tables/JSON/images,
and write reports/source_profile.json. Determine identity, image, mask, finding, label and missingness mappings
from source evidence only; write unresolved semantic questions to reports/questions.json. After decisions are known,
write reports/mapping.json, then implement and run deterministic processing code under scripts/. Produce candidate
annotation, image manifest, provenance and execution reports under candidate/. Run your own checks repeatedly until
counts, identities, paths, masks, labels and provenance are consistent. Write reports/preparation_report.json and
candidate/complete.json only after self-checks pass. Preserve raw strings, keep the source immutable, and never
claim medical correctness or publication. If a user decision is required, stop with reports/questions.json.
"""


class DatasetAgentState(TypedDict, total=False):
    context: dict
    result: dict


def _task(context: dict, workspace: Path) -> DatasetAgentTask:
    source = context["active_source"]
    session = load_session(workspace)
    conversation_id = session.get("conversation_id") or None
    return DatasetAgentTask(
        task_id=f"{context['run_id']}:{context['dataset_scope']}:{source['id']}:preparation",
        kind="preparation", source_root=context["source_root"],
        source_manifest=context["source_manifest"], workspace=str(workspace),
        objective=DATASET_WORKFLOW_PROMPT, required_outputs=[
            "reports/source_profile.json", "reports/mapping.json",
            "reports/preparation_report.json", "candidate/complete.json",
        ], feedback="", attempt=session.get("attempt", 1), conversation_id=conversation_id,
        max_iterations=context.get("coding_worker_max_iterations", 80),
        timeout_seconds=context.get("coding_worker_timeout_seconds", 3600),
        model=get_settings().llm.model, base_url=get_settings().llm.base_url,
        reasoning_effort=get_settings().llm.reasoning_effort, run_id=context["run_id"],
        dataset_id=context["dataset_scope"], source_id=source["id"], workflow_step="preparation",
    )


def run_dataset_agent(context: dict):
    context = dict(context)
    if not context.get("active_source"):
        sources = [Source.model_validate(item) for item in context.get("sources", [])]
        if not sources:
            return {"status": "blocked", "summary": "Dataset has no configured source", "updates": {}}
        source = sources[0]
        manifest = Path(context["bundle_root"]) / "source_manifests" / source.id
        inventory_source(source, manifest)
        context.update(source_scope=source.id, active_source=source.model_dump(),
                       source_root=str(Path(source.root).resolve()), source_manifest=str(manifest))
    source = context["active_source"]
    workspace = Path(context["run_directory"]) / "workers" / "openhands" / context["dataset_scope"] / source["id"]
    workspace.mkdir(parents=True, exist_ok=True)
    task = _task(context, workspace)
    feedback = ""
    outcome = None
    for attempt in range(task.attempt, task.attempt + 4):
        task = task.model_copy(update={"attempt": attempt, "feedback": feedback})
        write_session(workspace, task=task.model_dump(), status="running", step="preparation",
                      attempt=attempt)
        outcome = dataset_runtime(context).run(task)
        if outcome.status != "complete":
            break
        gate_error, payload = _candidate_gate(workspace)
        if not gate_error:
            updates = {"preparation_conversation_id": outcome.conversation_id,
                       "preparation_workspace": str(workspace),
                       "preparation_status": "complete", "preparation_candidate": payload}
            return {"status": "complete", "summary": "Dataset processing agent completed candidate preparation",
                    "updates": updates}
        feedback = gate_error
    updates = {"preparation_conversation_id": outcome.conversation_id if outcome else "",
               "preparation_workspace": str(workspace),
               "preparation_status": outcome.status if outcome else "failed"}
    return {"status": outcome.status if outcome and outcome.status != "complete" else "failed",
            "summary": (outcome.diagnostic or outcome.question if outcome else "") or feedback,
            "updates": updates}


def _candidate_gate(workspace: Path) -> tuple[str, dict | None]:
    """Minimal publication gate; detailed image/bundle checks remain downstream."""
    required = [workspace / "outputs" / name for name in (
        "reports/source_profile.json", "reports/mapping.json",
        "reports/preparation_report.json", "candidate/complete.json")]
    missing = [str(path.relative_to(workspace)) for path in required if not path.is_file()]
    if missing:
        error = {"status": "failed", "error": "missing_candidate_artifacts", "missing": missing}
    else:
        try:
            payload = json.loads(required[-1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            error = {"status": "failed", "error": "invalid_completion_marker", "detail": str(exc)}
        else:
            if not isinstance(payload, dict) or payload.get("status") not in {"complete", "submitted"}:
                error = {"status": "failed", "error": "invalid_completion_marker", "expected_status": "complete"}
            else:
                return "", payload
    report = workspace / "outputs" / "reports" / "final_gate_error.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
    return "Final candidate gate failed; read reports/final_gate_error.json and repair the workspace.", None


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



