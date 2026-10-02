"""annotation.v5 dataset registry."""

import json
from pathlib import Path
import re
from medbenchagent import settings
from medbenchagent.data.v5_catalog import read_v5_card, verify_v5_bundle
from medbenchagent.runtime.run_artifacts import atomic_write_json

DATASET_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class DatasetCatalogError(ValueError):
    pass


def validate_dataset_id(value):
    if not isinstance(value, str) or not DATASET_ID_PATTERN.fullmatch(value):
        raise DatasetCatalogError("unsafe id")
    return value


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (settings.PROJECT_ROOT / path).resolve()


def load_dataset_catalog(catalog_path=None):
    path = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"schema_version": "dataset-catalog.v5", "datasets": []})
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") not in {None, "dataset-catalog.v5"}:
        raise DatasetCatalogError("Only dataset-catalog.v5 is supported")
    entries = value.get("datasets", [])
    for entry in entries:
        validate_dataset_id(entry["dataset_id"])
        if "asset_manifest_path" not in entry:
            raise DatasetCatalogError("Catalog entries require asset_manifest_path")
    return entries


def dataset_by_id(datasets, dataset_id):
    for entry in datasets:
        if entry["dataset_id"] == dataset_id:
            if not entry["available"]:
                raise DatasetCatalogError("Dataset is disabled")
            return entry
    raise DatasetCatalogError(f"Unknown dataset id: {dataset_id}")


def load_prepared(entry):
    bundle = resolve_path(entry["annotation_path"]).parent
    verified = verify_v5_bundle(bundle)
    access = verified["ready"]["access"]
    frozen_card_path = resolve_path(entry.get("dataset_card_path", access["dataset_card_path"]))
    card = verified["card"] if frozen_card_path == resolve_path(access["dataset_card_path"]) else read_v5_card(frozen_card_path)
    if card.get("annotation_digest") != access["annotation_digest"] or card.get("asset_manifest_digest") != access["asset_manifest_digest"]:
        raise DatasetCatalogError("Frozen Dataset Card does not match annotation.v5 bundle")
    return card, access


def register_dataset(dataset_id, bundle, *, catalog_path=None):
    validate_dataset_id(dataset_id)
    bundle = Path(bundle).resolve()
    card = json.loads((bundle / "dataset_card.json").read_text(encoding="utf-8"))
    entry = {
        "dataset_id": dataset_id,
        "name": card["name"],
        "description": card["description"],
        "available": True,
        "annotation_path": str(bundle / "annotation.jsonl"),
        "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
        "profile_path": str(bundle / "dataset_profile.json"),
        "dataset_card_path": str(bundle / "dataset_card.json"),
    }
    target = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    entries = (
        [e for e in load_dataset_catalog(target) if e["dataset_id"] != dataset_id]
        if target.exists()
        else []
    )
    atomic_write_json(
        target, {"schema_version": "dataset-catalog.v5", "datasets": entries + [entry]}
    )
    return entry
