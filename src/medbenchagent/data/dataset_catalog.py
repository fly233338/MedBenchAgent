"""Validated dataset catalog loading and LLM-assisted recommendation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypedDict

from medbenchagent import settings
from medbenchagent.runtime.construction import event_operation
from medbenchagent.runtime.llm import complete_json

DATASET_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class DatasetCatalogError(ValueError):
    """Raised when the dataset catalog or a recommendation violates its contract."""


class DatasetEntry(TypedDict):
    id: str
    name: str
    description: str
    annotation_path: str
    available: bool


def load_dataset_catalog(catalog_path: str | Path | None = None) -> list[DatasetEntry]:
    """Load and deterministically validate the dataset catalog."""
    path = (
        Path(catalog_path)
        if catalog_path is not None
        else settings.get_settings().paths.dataset_catalog
    )
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DatasetCatalogError(f"Dataset catalog could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetCatalogError(f"Dataset catalog is not valid JSON: {path}") from exc
    if not isinstance(catalog, dict) or set(catalog) != {"datasets"}:
        raise DatasetCatalogError("Dataset catalog must contain exactly one datasets array.")
    raw_datasets = catalog["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise DatasetCatalogError("Dataset catalog must contain at least one dataset.")

    datasets: list[DatasetEntry] = []
    seen_ids: set[str] = set()
    required = {"id", "name", "description", "annotation_path", "available"}
    for index, raw in enumerate(raw_datasets):
        if not isinstance(raw, dict) or set(raw) != required:
            raise DatasetCatalogError(
                f"Dataset catalog entry {index} must contain exactly: {', '.join(sorted(required))}."
            )
        for field in ("id", "name", "description", "annotation_path"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise DatasetCatalogError(f"Dataset catalog entry {index} has an invalid {field}.")
        dataset_id = raw["id"].strip()
        if not DATASET_ID_PATTERN.fullmatch(dataset_id):
            raise DatasetCatalogError(
                f"Dataset catalog entry {index} has an unsafe id; use a lowercase stable slug."
            )
        if dataset_id in seen_ids:
            raise DatasetCatalogError(f"Dataset catalog contains duplicate id: {dataset_id}.")
        if not isinstance(raw["available"], bool):
            raise DatasetCatalogError(f"Dataset catalog entry {dataset_id} has an invalid available flag.")
        seen_ids.add(dataset_id)
        datasets.append(
            {
                "id": dataset_id,
                "name": raw["name"].strip(),
                "description": raw["description"].strip(),
                "annotation_path": raw["annotation_path"].strip(),
                "available": raw["available"],
            }
        )
    if not any(dataset["available"] for dataset in datasets):
        raise DatasetCatalogError("Dataset catalog must contain at least one available dataset.")
    return datasets


def public_dataset_entries(datasets: list[DatasetEntry]) -> list[dict[str, object]]:
    """Return catalog metadata safe for model prompts and user review."""
    return [
        {
            "id": dataset["id"],
            "name": dataset["name"],
            "description": dataset["description"],
            "available": dataset["available"],
        }
        for dataset in datasets
    ]


def dataset_by_id(datasets: list[DatasetEntry], dataset_id: str) -> DatasetEntry:
    """Return one available dataset entry by exact id."""
    for dataset in datasets:
        if dataset["id"] == dataset_id:
            if not dataset["available"]:
                raise DatasetCatalogError(f"Dataset is not available: {dataset_id}.")
            return dataset
    raise DatasetCatalogError(f"Unknown dataset id: {dataset_id}.")


def resolve_annotation_path(dataset: DatasetEntry) -> Path:
    """Resolve project-relative annotation paths without exposing them to the model."""
    path = Path(dataset["annotation_path"]).expanduser()
    return path.resolve() if path.is_absolute() else (settings.PROJECT_ROOT / path).resolve()


def _validate_recommendation(
    response: object, datasets: list[DatasetEntry]
) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(response, dict) or set(response) != {"dataset_id", "reason"}:
        return None, "recommendation must contain exactly dataset_id and reason"
    dataset_id = response.get("dataset_id")
    reason = response.get("reason")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None, "dataset_id must be a non-empty string"
    try:
        dataset_by_id(datasets, dataset_id.strip())
    except DatasetCatalogError as exc:
        return None, str(exc)
    if not isinstance(reason, str) or not reason.strip():
        return None, "reason must be a non-empty string"
    return {"dataset_id": dataset_id.strip(), "reason": reason.strip()}, None


def recommend_dataset(
    benchmark_info: dict[str, Any], datasets: list[DatasetEntry]
) -> dict[str, str]:
    """Recommend one available dataset and repair one invalid model response."""
    catalog = public_dataset_entries(datasets)
    prompt = (
        "Select the single available dataset that best matches the confirmed medical VLM benchmark. "
        "Return JSON only with exactly dataset_id and reason. Never select a dataset whose available "
        "field is false. Use only the supplied dataset ids.\n\n"
        f"BenchmarkInfo: {json.dumps(benchmark_info, ensure_ascii=False)}\n\n"
        f"Dataset catalog: {json.dumps(catalog, ensure_ascii=False)}"
    )
    response = complete_json(
        "Recommend one dataset for a medical VLM benchmark. Return JSON only.", prompt
    )
    recommendation, error = _validate_recommendation(response, datasets)
    if recommendation is not None:
        return recommendation
    with event_operation("repair"):
        repaired = complete_json(
            "Repair a medical dataset recommendation. Return JSON only.",
            f"{prompt}\n\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
            f"Contract error: {error}",
        )
    recommendation, error = _validate_recommendation(repaired, datasets)
    if recommendation is None:
        raise DatasetCatalogError(f"Dataset recommendation is invalid: {error}.")
    return recommendation
