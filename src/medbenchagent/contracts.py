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
    options: NotRequired[list[str]]


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
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


class BenchmarkGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Write validated benchmark items and return a compact result."""
