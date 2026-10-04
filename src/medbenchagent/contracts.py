"""Stable contracts passed from planning through benchmark generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict


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
    template_index: NotRequired[int]
    template_count: NotRequired[int]
    options: list[str] | None
    input_bindings: list[dict[str, object]]
    input_evidence_ids: list[str]
    reference_evidence_ids: list[str]
    answer: Any


@dataclass(frozen=True)
class GenerationResult:
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
