"""AuditAgent deterministic replay followed by sampled semantic screening."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from medbenchagent.agents.contracts import (
    AUDIT_AGENT,
    AuditAgentInputState,
    AuditAgentOutputState,
    AuditAgentState,
)
from medbenchagent.audit import audit_run
from medbenchagent.dataset_catalog import (
    dataset_by_id,
    load_dataset_catalog,
    resolve_annotation_path,
)
from medbenchagent.semantic_audit import semantic_audit_run


def _deterministic_audit_node(state: AuditAgentState) -> dict[str, Any]:
    dataset = dataset_by_id(load_dataset_catalog(), state["selected_dataset_id"])
    annotation_path = resolve_annotation_path(dataset)
    result = audit_run(
        raw_path=state["raw_output_path"],
        blueprint_path=state["working_blueprint_path"],
        provenance_path=state["provenance_output_path"],
        events_path=state["construction_events_path"],
        manifest_path=annotation_path.parent / "image_manifest.jsonl",
        annotation_path=annotation_path,
        annotation_sha256=str(state["dataset_access_manifest"]["annotation_sha256"]),
        output_directory=state["run_directory"],
        run_id=state["run_id"],
        seed=int(state.get("seed", 0)),
    )
    return {
        "audit_artifacts": {
            key.removesuffix("_path"): value
            for key, value in result.items()
            if key.endswith("_path")
        }
    }


def _semantic_audit_node(state: AuditAgentState) -> dict[str, Any]:
    artifacts = state["audit_artifacts"]
    result = semantic_audit_run(
        audited_path=artifacts["audited"],
        flagged_path=artifacts["flagged"],
        report_path=artifacts["report"],
        blueprint_path=state["working_blueprint_path"],
        events_path=state["construction_events_path"],
        output_directory=state["run_directory"],
        run_id=state["run_id"],
        dataset_id=state["selected_dataset_id"],
        seed=int(state.get("seed", 0)),
    )
    return {
        "audit_artifacts": artifacts
        | {
            key.removesuffix("_path"): value
            for key, value in result.items()
            if key.endswith("_path")
        }
    }


def compile_audit_agent(*, checkpointer: Any = None):
    """Build the role-level hard and sampled semantic AuditAgent."""
    graph = StateGraph(
        AuditAgentState,
        input_schema=AuditAgentInputState,
        output_schema=AuditAgentOutputState,
    )
    graph.add_node("deterministic_audit", _deterministic_audit_node)
    graph.add_node("semantic_audit", _semantic_audit_node)
    graph.add_edge(START, "deterministic_audit")
    graph.add_edge("deterministic_audit", "semantic_audit")
    graph.add_edge("semantic_audit", END)
    return graph.compile(checkpointer=checkpointer, name=AUDIT_AGENT)
