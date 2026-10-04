"""Canonical generation request contracts derived from Task Plan Leaves."""

from typing import Any, TypedDict


class GenerationTask(TypedDict, total=False):
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
    sample_count: int


class GenerationRequest(TypedDict):
    benchmark_info: dict[str, object]
    run_id: str
    dataset_id: str
    construction_events_path: str
    dataset_access_manifest: dict[str, object]
    tasks: dict[str, GenerationTask]
    target_sample_count: int
    seed: int
    max_tasks_per_image: int
    max_total_tasks_per_image: int | None
    question_temperature: float
    output_directory: str
