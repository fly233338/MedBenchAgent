"""Stable contracts passed from Task Planning through benchmark delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict


class LeafTaskContract(TypedDict, total=False):
    leaf_task_id: str
    name: str
    scenario_id: str
    question_templates: list[str]
    options: list[str] | None
    input: list[dict[str, Any]]
    target: str
    reference_rule: str
    resolver: dict[str, Any]
    output_spec: str
    task_paradigm: str
    capability_category: str
    eligible_count: int
    sample_count: NotRequired[int]


class BenchmarkItem(TypedDict, total=False):
    item_id: str
    id: str
    dataset_id: str
    dataset: str
    scenario_id: str
    leaf_task_id: str
    task_id: str
    case_id: str
    sampling_unit_id: str
    split: str
    view_ids: list[str]
    panel_id: str | None
    sampling_order: int
    capability_category: str
    task_paradigm: str
    output_spec: str
    question: str
    template_index: int
    template_count: int
    options: list[str] | None
    input_bindings: list[dict[str, Any]]
    input_assets: list[str]
    input_evidence_ids: list[str]
    reference_evidence_ids: list[str]
    answer: Any


@dataclass(frozen=True)
class GenerationResult:
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
