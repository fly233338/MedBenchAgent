"""Stable contracts passed from planning through benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict


class TaskGenerationSpec(TypedDict):
    task_id: str
    task_type: str
    response_format: str
    description: str
    question_templates: list[str]
    placeholder_bindings: dict[str, str]
    validation_rules: list[str]
    citations: list[dict[str, object]]
    options: NotRequired[list[str]]
    answer_field: NotRequired[str]
    answer_value_mapping: NotRequired[dict[str, str]]
    instance_path: NotRequired[str]
    target_text_field: NotRequired[str]
    bbox_field: NotRequired[str]
    bbox_format: NotRequired[str]


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
    dataset_access_manifest: dict[str, object]
    approved_task_plans: list[dict[str, object]]
    question_designs: list[dict[str, object]]
    task_specs: list[TaskGenerationSpec]
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
