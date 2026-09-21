"""Private generation state contracts."""

from __future__ import annotations
from typing import Any, TypedDict
from medbenchagent.runtime.construction import (
    GENERATION_AGENT as GENERATION_AGENT,
)
from medbenchagent.generation.contracts import GenerationRequest


class GenerationAgentInputState(TypedDict, total=False):
    benchmark_info: dict[str, Any]
    run_id: str
    selected_dataset_id: str
    working_blueprint_path: str
    construction_events_path: str
    dataset_access_manifest: dict[str, Any]
    seed: int
    max_tasks_per_image: int
    max_total_tasks_per_image: int | None
    question_temperature: float


class GenerationAgentState(GenerationAgentInputState, total=False):
    generation_request: GenerationRequest
    generation_target_counts: dict[str, int]
    generation_completed_counts: dict[str, int]
    generation_stage_index: int
    generation_stage: str
    generation_cursor: int
    generation_progress_path: str
    generation_rejections: dict[str, int]
    generation_exhausted: bool
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


class GenerationAgentOutputState(TypedDict):
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]
