"""AuditAgent wrapper around the deterministic benchmark audit."""

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


def _audit_node(state: AuditAgentState) -> dict[str, Any]:
    dataset = dataset_by_id(load_dataset_catalog(), state["selected_dataset_id"])
    annotation_path = resolve_annotation_path(dataset)
    result = audit_run(
        raw_path=state["raw_output_path"],
        blueprint_path=state["working_blueprint_path"],
        provenance_path=state["provenance_output_path"],
        events_path=state["construction_events_path"],
        manifest_path=annotation_path.parent / "image_manifest.jsonl",
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


def compile_audit_agent(*, checkpointer: Any = None):
    """Build the deterministic role-level AuditAgent."""
    graph = StateGraph(
        AuditAgentState,
        input_schema=AuditAgentInputState,
        output_schema=AuditAgentOutputState,
    )
    graph.add_node("audit", _audit_node)
    graph.add_edge(START, "audit")
    graph.add_edge("audit", END)
    return graph.compile(checkpointer=checkpointer, name=AUDIT_AGENT)
