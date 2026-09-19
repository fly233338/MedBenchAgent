import json
from pathlib import Path

import pytest

import medbenchagent.data.dataset_catalog as catalog


def _entry(dataset_id: str, *, available: bool = True) -> dict:
    return {
        "dataset_id": dataset_id,
        "name": dataset_id.title(),
        "description": f"Description for {dataset_id}.",
        "annotation_path": f"datasets/{dataset_id}/annotation.jsonl",
        "available": available,
        "image_manifest_path": f"datasets/{dataset_id}/image_manifest.jsonl",
        "profile_path": f"datasets/{dataset_id}/dataset_profile.json",
        "dataset_card_path": f"datasets/{dataset_id}/dataset_card.json",
    }


def _write_catalog(path: Path, datasets: list[dict]) -> Path:
    path.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    return path


def test_catalog_loads_and_resolves_project_relative_paths(tmp_path: Path, monkeypatch) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.json", [_entry("vindr-mammo")])
    monkeypatch.setattr("medbenchagent.settings.PROJECT_ROOT", tmp_path)

    datasets = catalog.load_dataset_catalog(catalog_path)

    assert datasets[0]["dataset_id"] == "vindr-mammo"
    assert catalog.resolve_annotation_path(datasets[0]) == (
        tmp_path / "datasets" / "vindr-mammo" / "annotation.jsonl"
    ).resolve()


@pytest.mark.parametrize(
    ("datasets", "expected"),
    [
        ([_entry("same"), _entry("same")], "duplicate id"),
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


def test_unprepared_entry_is_actionable(tmp_path):
    with pytest.raises(catalog.DatasetCatalogError, match="run prepare"):
        catalog.load_prepared(_entry("not-prepared"))


def test_empty_registry_can_be_initialized(tmp_path):
    assert catalog.load_dataset_catalog(_write_catalog(tmp_path / "registry.json", [])) == []
