"""annotation dataset registry."""

import json
from pathlib import Path
import re
from medbenchagent import settings
from medbenchagent.data.bundle_catalog import read_card, verify_bundle
from medbenchagent.runtime.run_artifacts import atomic_write_json

DATASET_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class DatasetCatalogError(ValueError):
    pass


def validate_dataset_id(value):
    if not isinstance(value, str) or not DATASET_ID_PATTERN.fullmatch(value):
        raise DatasetCatalogError("unsafe id")
    return value


def normalize_dataset_id(value: str) -> str:
    """Derive a safe dynamic ID from a user-facing dataset name or directory."""
    if not isinstance(value, str):
        raise DatasetCatalogError("dataset name must be a string")
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not normalized:
        raise DatasetCatalogError("dataset name cannot produce a safe id")
    return validate_dataset_id(normalized)


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (settings.PROJECT_ROOT / path).resolve()


def prepared_bundle_entry(dataset_id, *, prepared_root=None):
    """Build an entry for one valid direct Prepared/<dataset_id> bundle."""
    validate_dataset_id(dataset_id)
    root = Path(prepared_root or settings.get_settings().paths.prepared_datasets_root).resolve()
    bundle = root / dataset_id
    ready = bundle / "ready.json"
    if not ready.is_file():
        raise DatasetCatalogError(f"Prepared bundle is unavailable: {bundle}")
    document = json.loads(ready.read_text(encoding="utf-8"))
    access = document.get("access", {})
    cases = Path(access.get("cases_path", ""))
    if cases.parent.resolve() != bundle or not cases.is_file():
        raise DatasetCatalogError("Prepared bundle cases_path must point to its direct bundle")
    card = read_card(access.get("dataset_card_path", bundle / "dataset_card.json"))
    if card.get("dataset_id") != dataset_id:
        raise DatasetCatalogError("Prepared bundle Card dataset_id does not match its directory")
    value = {
        "dataset_id": dataset_id,
        "name": card["name"],
        "description": card["description"],
        "available": True,
        "cases_path": str(cases),
        "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
        "profile_path": str(bundle / "dataset_profile.json"),
        "dataset_card_path": str(bundle / "dataset_card.json"),
    }
    for name in ("presentation", "split", "sampling"):
        path_key = f"{name}_manifest_path"
        digest_key = f"{name}_manifest_digest"
        if access.get(path_key):
            value[path_key] = access[path_key]
            value[digest_key] = access.get(digest_key)
    return value


def load_dataset_catalog(catalog_path=None):
    path = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"datasets": []})
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
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
    bundle = resolve_path(entry["cases_path"]).parent
    verified = verify_bundle(bundle)
    access = verified["ready"]["access"]
    frozen_card_path = resolve_path(entry.get("dataset_card_path", access["dataset_card_path"]))
    card = verified["card"] if frozen_card_path == resolve_path(access["dataset_card_path"]) else read_card(frozen_card_path)
    if card.get("annotation_digest") != access["annotation_digest"] or card.get("asset_manifest_digest") != access["asset_manifest_digest"]:
        raise DatasetCatalogError("Frozen Dataset Card does not match annotation bundle")
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
        "cases_path": str(bundle / "cases.jsonl"),
        "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
        "profile_path": str(bundle / "dataset_profile.json"),
        "dataset_card_path": str(bundle / "dataset_card.json"),
    }
    ready = json.loads((bundle / "ready.json").read_text(encoding="utf-8")) if (bundle / "ready.json").is_file() else {}
    access = ready.get("access", {})
    for name in ("presentation", "split", "sampling"):
        path_key = f"{name}_manifest_path"
        digest_key = f"{name}_manifest_digest"
        if access.get(path_key):
            entry[path_key] = access[path_key]
            entry[digest_key] = access.get(digest_key)
    target = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    entries = (
        [e for e in load_dataset_catalog(target) if e["dataset_id"] != dataset_id]
        if target.exists()
        else []
    )
    atomic_write_json(
        target, {"datasets": entries + [entry]}
    )
    return entry

