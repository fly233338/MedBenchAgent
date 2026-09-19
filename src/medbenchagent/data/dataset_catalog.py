"""Registry of independently prepared, immutable dataset versions."""

import json
from pathlib import Path
import re
from medbenchagent import settings
from medbenchagent.data.dataset_card import DatasetCard, validate_semantics
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json

DATASET_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
PATH_FIELDS = ("annotation_path", "image_manifest_path", "profile_path", "dataset_card_path")


class DatasetCatalogError(ValueError):
    """Invalid registry entry or unavailable prepared asset."""


def validate_dataset_id(value):
    if not isinstance(value, str) or not DATASET_ID_PATTERN.fullmatch(value):
        raise DatasetCatalogError("unsafe id; use a lowercase stable slug")
    return value


def load_dataset_catalog(catalog_path=None):
    path = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetCatalogError(f"Cannot read dataset catalog: {path}") from exc
    if not isinstance(catalog, dict) or set(catalog) != {"datasets"} or not isinstance(catalog["datasets"], list):
        raise DatasetCatalogError("Catalog requires one datasets array")
    seen = set()
    required = {"dataset_id", "name", "description", "available", *PATH_FIELDS}
    for entry in catalog["datasets"]:
        if not isinstance(entry, dict) or set(entry) != required:
            raise DatasetCatalogError(f"Catalog entry must contain exactly {sorted(required)}")
        identifier = validate_dataset_id(entry["dataset_id"])
        if identifier in seen:
            raise DatasetCatalogError(f"duplicate id: {identifier}")
        seen.add(identifier)
        if type(entry["available"]) is not bool:
            raise DatasetCatalogError("invalid available flag")
        if any(not isinstance(entry[key], str) or not entry[key].strip() for key in required - {"available"}):
            raise DatasetCatalogError("Catalog text and paths must be nonempty")
    return catalog["datasets"]


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (settings.PROJECT_ROOT / path).resolve()


def resolve_annotation_path(entry):
    return resolve_path(entry["annotation_path"])


def dataset_by_id(datasets, dataset_id):
    for entry in datasets:
        if entry["dataset_id"] == dataset_id:
            if not entry["available"]:
                raise DatasetCatalogError(f"Dataset is not available: {dataset_id}")
            return entry
    raise DatasetCatalogError(f"Unknown dataset id: {dataset_id}")


def load_prepared(entry):
    paths = {key: resolve_path(entry[key]) for key in PATH_FIELDS}
    for key, path in paths.items():
        if not path.is_file():
            raise DatasetCatalogError(f"{entry['dataset_id']} lacks {key}; run prepare first: {path}")
    ready_path = paths["annotation_path"].parent / "ready.json"
    if not ready_path.is_file():
        raise DatasetCatalogError("Dataset bundle is not published; run prepare first")
    access = json.loads(ready_path.read_text(encoding="utf-8"))["access"]
    card = DatasetCard.model_validate_json(paths["dataset_card_path"].read_text(encoding="utf-8"))
    profile = json.loads(paths["profile_path"].read_text(encoding="utf-8"))
    if (card.dataset_id != entry["dataset_id"] or
        Path(access["annotation_path"]).resolve() != paths["annotation_path"] or
        Path(access["profile_path"]).resolve() != paths["profile_path"] or
        not Path(access["duckdb_path"]).is_file() or
        card.annotation_digest != access["annotation_digest"] or
        card.annotation_digest != digest_file(paths["annotation_path"]) or
        card.profile_digest != digest_file(paths["profile_path"]) or
        card.image_manifest_digest != digest_file(paths["image_manifest_path"]) or
        card.total_images != profile["total_images"] or card.fields != profile["fields"] or
        card.structures != profile["structures"] or card.summary != profile["summary"]):
        raise DatasetCatalogError("Prepared dataset/Card facts changed; prepare a new version")
    blueprint = settings.get_settings().paths.blueprint_template
    if card.blueprint_digest != digest_file(blueprint):
        raise DatasetCatalogError("Card Blueprint changed; prepare a new dataset version")
    validate_semantics(card.semantics, profile, json.loads(blueprint.read_text(encoding="utf-8"))["concepts"])
    return card, access


def register_dataset(dataset_id, bundle, *, catalog_path=None):
    validate_dataset_id(dataset_id)
    bundle = Path(bundle).resolve()
    ready = json.loads((bundle / "ready.json").read_text(encoding="utf-8"))
    card = DatasetCard.model_validate_json((bundle / "dataset_card.json").read_text(encoding="utf-8"))
    entry = {"dataset_id": dataset_id, "name": card.name, "description": card.semantics.description,
             "available": True, "annotation_path": str(bundle / "annotation.jsonl"),
             "image_manifest_path": str(bundle / "image_manifest.jsonl"),
             "profile_path": ready["access"]["profile_path"], "dataset_card_path": str(bundle / "dataset_card.json")}
    load_prepared(entry)
    target = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    entries = load_dataset_catalog(target) if target.is_file() else []
    entries = [e for e in entries if e["dataset_id"] != dataset_id] + [entry]
    atomic_write_json(target, {"datasets": sorted(entries, key=lambda e: e["dataset_id"])})
    return entry
