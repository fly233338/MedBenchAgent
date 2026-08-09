"""Stable contracts shared by benchmark loading, inference, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
class ImageEntry:
    image_id: str
    png_path: Path
    relative_png_path: str
    width: int
    height: int
    sha256: str


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
    benchmark_sha256: str
    benchmark_manifest_sha256: str
    image_manifest_sha256: str
