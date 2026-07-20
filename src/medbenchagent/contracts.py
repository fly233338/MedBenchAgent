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
    supporting_fields: list[str]
    dataset_support_reason: str
    question_templates: list[str]
    generation_logic: str
    medical_task_description: str
    citations: list[dict[str, object]]
    options: NotRequired[list[str]]


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
    dataset_access_manifest: dict[str, object]
    tasks: dict[str, BlueprintTask]
    target_sample_count: int
    seed: int
    balance_answer_positions: bool
    output_directory: str


class BenchmarkItem(TypedDict):
    id: str
    task_id: str
    task_type: str
    response_format: str
    image_ids: list[str]
    image_paths: list[str]
    question: str
    options: list[str] | None
    answer: str | list[float]
    answer_value: NotRequired[str]
    source_refs: list[dict[str, object]]
    medical_sources: list[dict[str, object]]
    metadata: NotRequired[dict[str, object]]


@dataclass(frozen=True)
class GenerationResult:
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]


class BenchmarkGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Write validated benchmark items and return a compact result."""
