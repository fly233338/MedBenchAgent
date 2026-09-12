"""Strict loading of generated benchmark items and their audited image mapping."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from medbenchagent.data.image_assets import validate_image_asset

from medbenchagent.evaluation.contracts import (
    ImageEntry,
    InferenceSample,
    LoadedBenchmark,
    RESPONSE_FORMATS,
    TaskSpec,
)
from medbenchagent.data.image_assets import ImageManifestError, load_image_manifest
from medbenchagent.metrics import metrics_by_response_format


METRICS_BY_FORMAT = metrics_by_response_format()
SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG"})


class EvaluationDataError(ValueError):
    """Raised when benchmark evaluation inputs violate their stable contracts."""




def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationDataError(f"{label} could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise EvaluationDataError(f"{label} must be a JSON object.")
    return payload


def _manifest_tasks(manifest: dict[str, Any]) -> dict[str, TaskSpec]:
    if manifest.get("schema_version") != "benchmark-manifest.v2":
        raise EvaluationDataError("Unsupported benchmark manifest schema_version.")
    raw_tasks = manifest.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise EvaluationDataError("Benchmark manifest must contain a non-empty tasks object.")
    tasks: dict[str, TaskSpec] = {}
    for task_id, raw in raw_tasks.items():
        if not isinstance(task_id, str) or not task_id or not isinstance(raw, dict):
            raise EvaluationDataError("Benchmark manifest contains an invalid task.")
        category = raw.get("category")
        response_format = raw.get("response_format")
        metrics = raw.get("metrics")
        parent_task_id = raw.get("parent_task_id")
        if (
            not isinstance(category, str)
            or not category
            or response_format not in METRICS_BY_FORMAT
            or not isinstance(metrics, list)
            or len(metrics) != 1
            or metrics[0] not in METRICS_BY_FORMAT[response_format]
            or (parent_task_id is not None and not isinstance(parent_task_id, str))
        ):
            raise EvaluationDataError(f"{task_id}: unsupported response format or metric.")
        tasks[task_id] = TaskSpec(
            task_id,
            category,
            response_format,
            metrics[0],
            parent_task_id,
        )
    return tasks




def _load_image_manifest(path: Path) -> dict[str, ImageEntry]:
    try:
        return load_image_manifest(path)
    except ImageManifestError as exc:
        raise EvaluationDataError(str(exc)) from exc


def _validate_item(item: dict[str, Any], task: TaskSpec, image: ImageEntry) -> None:
    required_strings = ("id", "task_id", "dataset", "image_id", "image_path", "question")
    if any(not isinstance(item.get(field), str) or not item[field].strip() for field in required_strings):
        raise EvaluationDataError("Benchmark item has a missing or invalid required string field.")
    if task.response_format == "letter":
        options, answer = item.get("options"), item.get("answer")
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= 26
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len(set(options)) != len(options)
            or not isinstance(answer, str)
            or len(answer) != 1
            or not answer.isalpha()
            or ord(answer.upper()) - ord("A") not in range(len(options))
        ):
            raise EvaluationDataError(f"{item['id']}: invalid letter response contract.")
    elif task.response_format == "free_text":
        if item.get("options") is not None or not isinstance(item.get("answer"), str) or not item["answer"]:
            raise EvaluationDataError(f"{item['id']}: invalid free-text response contract.")
    else:
        box = item.get("answer")
        if (
            item.get("options") is not None
            or not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
            or not (0 <= box[0] < box[2] <= image.width)
            or not (0 <= box[1] < box[3] <= image.height)
        ):
            raise EvaluationDataError(f"{item['id']}: invalid bbox response contract.")


def _validate_image(image: ImageEntry) -> None:
    errors = validate_image_asset(
        {"png_path": image.png_path.name, "width": image.width, "height": image.height},
        image.png_path.parent,
        check_uniform=False,
        supported_formats=SUPPORTED_IMAGE_FORMATS,
    )
    if errors:
        messages = {
            "image_missing": f"image is missing: {image.png_path}",
            "image_decode": "image could not be decoded.",
            "image_format": "image format or dimensions do not match image manifest.",
            "image_dimensions": "image format or dimensions do not match image manifest.",
        }
        raise EvaluationDataError(f"{image.image_id}: {messages.get(errors[0], errors[0])}")


def load_benchmark(
    benchmark_path: str | Path,
    benchmark_manifest_path: str | Path,
    image_manifest_path: str | Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    task_types: tuple[str, ...] | None = None,
) -> LoadedBenchmark:
    """Load and validate one generated benchmark for a specific inference rank."""
    benchmark_file = Path(benchmark_path)
    manifest_file = Path(benchmark_manifest_path)
    image_manifest_file = Path(image_manifest_path)
    manifest = _read_json(manifest_file, "Benchmark manifest")
    tasks = _manifest_tasks(manifest)
    selected_types = tuple(RESPONSE_FORMATS if task_types is None else task_types)
    if (
        not selected_types
        or len(set(selected_types)) != len(selected_types)
        or any(task_type not in RESPONSE_FORMATS for task_type in selected_types)
    ):
        raise EvaluationDataError("Task types must be unique supported response formats.")
    selected_type_set = set(selected_types)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("dataset_id"), str):
        raise EvaluationDataError("Benchmark manifest has an invalid dataset object.")
    benchmark_info = manifest.get("benchmark_info")
    if not isinstance(benchmark_info, dict) or benchmark_info.get("language") not in {"en", "zh"}:
        raise EvaluationDataError("Benchmark manifest has an invalid benchmark_info language.")
    images = _load_image_manifest(image_manifest_file)

    try:
        lines = benchmark_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationDataError(f"Benchmark JSONL could not be read: {benchmark_file}") from exc
    if not lines:
        raise EvaluationDataError("Benchmark JSONL must not be empty.")
    samples: list[InferenceSample] = []
    seen_item_ids: set[str] = set()
    validated_images: set[str] = set()
    for index, line in enumerate(lines):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"Benchmark line {index + 1} is not valid JSON.") from exc
        if not isinstance(item, dict):
            raise EvaluationDataError(f"Benchmark line {index + 1} must be a JSON object.")
        item_id, task_id, image_id = item.get("id"), item.get("task_id"), item.get("image_id")
        if not isinstance(item_id, str) or item_id in seen_item_ids:
            raise EvaluationDataError(f"Benchmark line {index + 1} has a duplicate or invalid id.")
        if task_id not in tasks:
            raise EvaluationDataError(f"{item_id}: unknown task_id.")
        if item.get("dataset") != dataset["dataset_id"]:
            raise EvaluationDataError(f"{item_id}: dataset does not match benchmark manifest.")
        if image_id not in images:
            raise EvaluationDataError(f"{item_id}: image_id is missing from image manifest.")
        task, image = tasks[task_id], images[image_id]
        _validate_item(item, task, image)
        if task.response_format not in selected_type_set:
            seen_item_ids.add(item_id)
            continue
        selected_index = len(samples)
        if selected_index % world_size == rank and image_id not in validated_images:
            _validate_image(image)
            validated_images.add(image_id)
        seen_item_ids.add(item_id)
        samples.append(InferenceSample(selected_index, item, task, image))

    present_types = {sample.task.response_format for sample in samples}
    missing_types = [task_type for task_type in selected_types if task_type not in present_types]
    if task_types is not None and missing_types:
        raise EvaluationDataError(
            "Benchmark contains no samples for requested task types: " + ", ".join(missing_types)
        )

    benchmark_id = manifest.get("benchmark_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise EvaluationDataError("Benchmark manifest has an invalid benchmark_id.")
    return LoadedBenchmark(
        benchmark_info=benchmark_info,
        benchmark_id=benchmark_id,
        dataset_id=dataset["dataset_id"],
        samples=samples,
    )
