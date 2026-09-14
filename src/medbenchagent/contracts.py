"""Stable contracts passed from planning through benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

class BlueprintTask(TypedDict):
    concept_id: str
    description: str
    category: str
    protocol: dict[str, object]
    evidence: list[dict[str, object]]
    sample_count: NotRequired[int]
    question_templates: NotRequired[list[str]]


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
    run_id: str
    dataset_id: str
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
