"""Stable contracts shared by benchmark loading, inference, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from medbenchagent.image_assets import ImageEntry as ImageEntry
from typing import Any


RESPONSE_FORMATS = ("letter", "bbox", "free_text")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    response_format: str
    metric: str
    parent_task_id: str | None = None


@dataclass(frozen=True)
class InferenceSample:
    index: int
    item: dict[str, Any]
    task: TaskSpec
    image: ImageEntry


@dataclass(frozen=True)
class ModelOutput:
    raw_text: str
    coordinate_size: tuple[int, int] | None = None
    answer_text: str | None = None


@dataclass(frozen=True)
class LoadedBenchmark:
    benchmark_info: dict[str, Any]
    benchmark_id: str
    dataset_id: str
    samples: list[InferenceSample]
