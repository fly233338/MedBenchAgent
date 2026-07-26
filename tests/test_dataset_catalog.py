import json
from pathlib import Path

import pytest

import medbenchagent.dataset_catalog as catalog


def _entry(dataset_id: str, *, available: bool = True) -> dict:
    return {
        "id": dataset_id,
        "name": dataset_id.title(),
        "description": f"Description for {dataset_id}.",
        "annotation_path": f"datasets/{dataset_id}/annotation.jsonl",
        "available": available,
    }


def _write_catalog(path: Path, datasets: list[dict]) -> Path:
    path.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    return path


def test_catalog_loads_and_resolves_project_relative_paths(tmp_path: Path, monkeypatch) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json", [_entry("vindr-mammo")])
    monkeypatch.setattr(catalog, "PROJECT_ROOT", tmp_path)

    datasets = catalog.load_dataset_catalog(catalog_path)

    assert datasets[0]["id"] == "vindr-mammo"
    assert catalog.resolve_annotation_path(datasets[0]) == (
        tmp_path / "datasets" / "vindr-mammo" / "annotation.jsonl"
    ).resolve()


@pytest.mark.parametrize(
    ("datasets", "expected"),
    [
        ([_entry("same"), _entry("same")], "duplicate id"),
        ([_entry("future", available=False)], "at least one available"),
        ([{**_entry("vindr-mammo"), "available": "yes"}], "invalid available"),
        ([_entry("unsafe id")], "unsafe id"),
    ],
)
def test_catalog_rejects_invalid_entries(
    tmp_path: Path, datasets: list[dict], expected: str
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json", datasets)

    with pytest.raises(catalog.DatasetCatalogError, match=expected):
        catalog.load_dataset_catalog(catalog_path)


def test_recommendation_repairs_unavailable_selection_without_exposing_paths(monkeypatch) -> None:
    datasets = [_entry("vindr-mammo"), _entry("future", available=False)]
    responses = iter(
        [
            {"dataset_id": "future", "reason": "Looks relevant."},
            {"dataset_id": "vindr-mammo", "reason": "Available and relevant."},
        ]
    )
    prompts: list[str] = []

    def complete(_system: str, prompt: str) -> dict:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(catalog, "complete_json", complete)

    result = catalog.recommend_dataset({"modality": "mammography"}, datasets)

    assert result == {"dataset_id": "vindr-mammo", "reason": "Available and relevant."}
    assert all("annotation_path" not in prompt for prompt in prompts)
    assert all("datasets/future" not in prompt for prompt in prompts)
