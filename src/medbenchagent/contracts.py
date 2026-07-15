"""Stable contracts passed from planning to a future generator."""

from __future__ import annotations

from typing import NotRequired, Protocol, TypedDict


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
    dataset_access_manifest: dict[str, object]
    approved_task_plans: list[dict[str, object]]
    question_designs: list[dict[str, object]]
    target_sample_count: int
    seed: int


class BenchmarkItem(TypedDict):
    id: str
    task_id: str
    image_ids: list[str]
    image_paths: list[str]
    question: str
    options: list[str] | None
    answer: str
    answer_value: str
    source_refs: list[dict[str, object]]
    medical_sources: list[dict[str, object]]
    metadata: NotRequired[dict[str, object]]


class BenchmarkGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> list[BenchmarkItem]:
        """Generate validated benchmark items from an approved planning request."""
