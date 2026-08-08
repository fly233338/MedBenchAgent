"""Deterministic root orchestration for the five role-specialized Agents."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, MessagesState, StateGraph

from medbenchagent.agents import (
    compile_audit_agent,
    compile_capability_agent,
    compile_clinical_design_agent,
    compile_generation_agent,
    compile_specification_agent,
)
from medbenchagent.agents.contracts import (
    AUDIT_AGENT,
    CAPABILITY_AGENT,
    CLINICAL_DESIGN_AGENT,
    GENERATION_AGENT,
    SPECIFICATION_AGENT,
)
from medbenchagent.construction import record_event
from medbenchagent.dataset_catalog import (
    dataset_by_id,
    load_dataset_catalog,
    resolve_annotation_path,
)
from medbenchagent.run_artifacts import initialize_run, latest_versioned_json
from medbenchagent.run_export import finalize_run
from medbenchagent.settings import get_settings

__all__ = [
    "BenchmarkInputState",
    "BenchmarkOutputState",
    "BenchmarkState",
    "benchmark_graph",
    "compile_benchmark_graph",
]


class BenchmarkInputState(TypedDict, total=False):
    """Values accepted when starting or resuming the root graph."""

    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    question_temperature: float


class BenchmarkState(MessagesState, total=False):
    """Cross-Agent values and stable artifact paths checkpointed by the root graph."""

    user_requirement: str
    seed: int
    max_tasks_per_image: int
    question_temperature: float
    benchmark_info: dict[str, Any]
    run_id: str
    run_directory: str
    construction_events_path: str
    working_blueprint_path: str
    selected_dataset_id: str
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    task_screening_report_path: str
    selected_material_files: list[str]
    raw_output_path: str
    provenance_output_path: str
    rag_designs_path: str
    construction_report_path: str
    manifest_path: str
    benchmark_id: str
    audit_artifacts: dict[str, str]
    generation_shortfalls: dict[str, int]
    item_count: int
    task_counts: dict[str, int]


class BenchmarkOutputState(TypedDict):
    """Stable result returned by the root benchmark graph."""

    benchmark_info: dict[str, Any]
    run_id: str
    selected_dataset_id: str
    working_blueprint_path: str
    task_screening_report_path: str
    raw_output_path: str
    provenance_output_path: str
    rag_designs_path: str
    construction_events_path: str
    construction_report_path: str
    manifest_path: str
    benchmark_id: str
    audit_artifacts: dict[str, str]
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


def _run_initialize_node(state: BenchmarkState) -> dict[str, Any]:
    """Create persistent run artifacts before any LLM-backed Agent."""
    if state.get("run_id"):
        return {}
    overrides = {
        key: state[key]
        for key in ("seed", "max_tasks_per_image", "question_temperature")
        if key in state
    }
    return initialize_run(settings=get_settings(), graph_overrides=overrides)


def _rag_design_export_node(state: BenchmarkState) -> dict[str, Any]:
    artifact = latest_versioned_json(state["run_directory"], "rag_designs")
    if artifact is None:
        raise ValueError("RAG design review completed without a persisted design artifact.")
    return {"rag_designs_path": str(artifact["path"])}


def _generation_summary_node(state: BenchmarkState) -> dict[str, Any]:
    record_event(
        state["construction_events_path"],
        "generation_summary",
        run_id=state["run_id"],
        dataset_id=state["selected_dataset_id"],
        agent=GENERATION_AGENT,
        operation="summary",
        status="complete",
        task_counts=state["task_counts"],
        shortfalls=state["generation_shortfalls"],
    )
    return {}


def _run_finalize_node(state: BenchmarkState) -> dict[str, Any]:
    dataset = dataset_by_id(load_dataset_catalog(), state["selected_dataset_id"])
    identifier, report_path, manifest_path = finalize_run(
        run_directory=state["run_directory"],
        run_id=state["run_id"],
        blueprint_path=state["working_blueprint_path"],
        raw_output_path=state["raw_output_path"],
        provenance_output_path=state["provenance_output_path"],
        rag_designs_path=state["rag_designs_path"],
        construction_events_path=state["construction_events_path"],
        dataset_catalog_entry=dataset,
        dataset_profile=state["dataset_profile"],
        dataset_access_manifest=state["dataset_access_manifest"],
        image_manifest_path=resolve_annotation_path(dataset).parent / "image_manifest.jsonl",
        audit_artifacts=state["audit_artifacts"],
    )
    return {
        "benchmark_id": identifier,
        "construction_report_path": report_path,
        "manifest_path": manifest_path,
    }


def compile_benchmark_graph(*, checkpointer: Any = None):
    """Build the deterministic five-Agent benchmark workflow."""
    graph = StateGraph(
        BenchmarkState,
        input_schema=BenchmarkInputState,
        output_schema=BenchmarkOutputState,
    )
    graph.add_node("run_initialize", _run_initialize_node)
    graph.add_node(SPECIFICATION_AGENT, compile_specification_agent())
    graph.add_node(CAPABILITY_AGENT, compile_capability_agent())
    graph.add_node(CLINICAL_DESIGN_AGENT, compile_clinical_design_agent())
    graph.add_node("rag_design_export", _rag_design_export_node)
    graph.add_node(GENERATION_AGENT, compile_generation_agent())
    graph.add_node("generation_summary", _generation_summary_node)
    graph.add_node(AUDIT_AGENT, compile_audit_agent())
    graph.add_node("run_finalize", _run_finalize_node)
    graph.add_edge(START, "run_initialize")
    graph.add_edge("run_initialize", SPECIFICATION_AGENT)
    graph.add_edge(SPECIFICATION_AGENT, CAPABILITY_AGENT)
    graph.add_edge(CAPABILITY_AGENT, CLINICAL_DESIGN_AGENT)
    graph.add_edge(CLINICAL_DESIGN_AGENT, "rag_design_export")
    graph.add_edge("rag_design_export", GENERATION_AGENT)
    graph.add_edge(GENERATION_AGENT, "generation_summary")
    graph.add_edge("generation_summary", AUDIT_AGENT)
    graph.add_edge(AUDIT_AGENT, "run_finalize")
    graph.add_edge("run_finalize", END)
    return graph.compile(checkpointer=checkpointer)


benchmark_graph = compile_benchmark_graph()
