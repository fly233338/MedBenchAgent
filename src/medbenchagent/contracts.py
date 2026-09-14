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
