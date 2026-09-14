"""Private generation inputs assembled from a Plan and its separate template artifact."""

from typing import TypedDict

from medbenchagent.contracts import BlueprintTask


class GenerationTask(BlueprintTask):
    question_templates: list[str]


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
    question_temperature: float
    output_directory: str
