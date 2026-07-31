"""Stable contracts passed from planning through benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict


class BlueprintTask(TypedDict):
    task_type: str
    response_format: str
    description: str
    require: str
    metric: str
    category: str
    supporting_fields: list[str]
    dataset_support_reason: str
    question_templates: list[str]
    generation_logic: str
    medical_task_description: str
    citations: list[dict[str, object]]
    sample_count: int | None
    parent_task_id: NotRequired[str]
    question_type_id: NotRequired[str]
    options: NotRequired[list[str]]
    answer_field: NotRequired[str]
    answer_value_mapping: NotRequired[list[dict[str, object]]]


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
    run_id: str
    dataset_id: str
    blueprint_sha256: str
    construction_events_path: str
    dataset_access_manifest: dict[str, object]
    tasks: dict[str, BlueprintTask]
    target_sample_count: int
    seed: int
    max_tasks_per_image: int
    question_temperature: float
    output_directory: str


class BenchmarkItem(TypedDict):
    id: str
    task_id: str
    dataset: str
    image_id: str
    image_path: str
    question: str
    options: list[str] | None
    answer: str | list[float]


@dataclass(frozen=True)
class GenerationResult:
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
    provenance_output_path: str = ""


class BenchmarkGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Write validated benchmark items and return a compact result."""
