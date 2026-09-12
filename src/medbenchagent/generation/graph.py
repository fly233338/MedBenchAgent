"""LangGraph subgraph for resumable benchmark-item generation."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from medbenchagent.contracts import GenerationRequest
from medbenchagent.generation.engine import (
    advance_generation,
    finalize_generation,
    generation_done,
    initialize_generation,
)


class GenerationInputState(TypedDict):
    """The fixed planning snapshot accepted by the generation subgraph."""

    generation_request: GenerationRequest


class GenerationGraphState(TypedDict, total=False):
    """Private, checkpointed state for the record-by-record generation loop."""

    generation_request: GenerationRequest
    generation_target_counts: dict[str, int]
    generation_completed_counts: dict[str, int]
    generation_stage_index: int
    generation_stage: str
    generation_cursor: int
    generation_progress_path: str
    generation_rejections: dict[str, int]
    generation_exhausted: bool
    generation_shortfalls: dict[str, int]
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]


class GenerationOutputState(TypedDict):
    """Stable result returned from the generation subgraph to the root graph."""

    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


def _emit_custom_event(event: dict[str, Any]) -> None:
    try:
        get_stream_writer()(event)
    except RuntimeError:
        return


def _generation_initialize_node(state: GenerationGraphState) -> dict[str, Any]:
    return initialize_generation(state["generation_request"])


def _generation_step_node(state: GenerationGraphState) -> dict[str, Any]:
    update = advance_generation(state["generation_request"], state)
    completed_counts = update.get(
        "generation_completed_counts", state["generation_completed_counts"]
    )
    _emit_custom_event(
        {
            "kind": "generation_progress",
            "stage": update.get("generation_stage", state.get("generation_stage", "")),
            "cursor": update.get("generation_cursor", state["generation_cursor"]),
            "completed_counts": completed_counts,
            "target_counts": state["generation_target_counts"],
            "rejections": update.get("generation_rejections", state["generation_rejections"]),
            "exhausted": update.get("generation_exhausted", state["generation_exhausted"]),
        }
    )
    return update


def _generation_finalize_node(state: GenerationGraphState) -> dict[str, Any]:
    result = finalize_generation(state["generation_request"])
    targets = state["generation_target_counts"]
    shortfalls = {
        task_id: max(0, target - result.task_counts.get(task_id, 0))
        for task_id, target in targets.items()
    }
    return {
        "raw_output_path": result.raw_output_path,
        "generation_completed_counts": result.task_counts,
        "generation_shortfalls": shortfalls,
        "item_count": result.item_count,
        "task_counts": result.task_counts,
    }


def compile_generation_subgraph(*, checkpointer: Any = None):
    """Build the checkpointable record-by-record generation loop."""
    graph = StateGraph(
        GenerationGraphState,
        input_schema=GenerationInputState,
        output_schema=GenerationOutputState,
    )
    graph.add_node("generation_initialize", _generation_initialize_node)
    graph.add_node("generation_step", _generation_step_node)
    graph.add_node("generation_finalize", _generation_finalize_node)
    graph.add_edge(START, "generation_initialize")
    graph.add_conditional_edges(
        "generation_initialize",
        lambda state: "generation_finalize" if generation_done(state) else "generation_step",
        {
            "generation_step": "generation_step",
            "generation_finalize": "generation_finalize",
        },
    )
    graph.add_conditional_edges(
        "generation_step",
        lambda state: "generation_finalize" if generation_done(state) else "generation_step",
        {
            "generation_step": "generation_step",
            "generation_finalize": "generation_finalize",
        },
    )
    graph.add_edge("generation_finalize", END)
    return graph.compile(checkpointer=checkpointer, name="generate_raw_benchmark")
