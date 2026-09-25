"""Registry of independently prepared, immutable dataset versions."""

import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4
from medbenchagent import settings
from medbenchagent.data.dataset_card import build_card, read_card, validate_semantics
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
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"datasets": []}, stream, indent=2)
                stream.write("\n")
            try:
                os.rename(temporary, path)
            except FileExistsError:
                pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
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


def load_prepared_assets(entry):
    """Validate the data/cache independently of the Card, including during v1 migration."""
    paths = {key: resolve_path(entry[key]) for key in PATH_FIELDS if key != "dataset_card_path"}
    for key, path in paths.items():
        if not path.is_file():
            raise DatasetCatalogError(f"{entry['dataset_id']} lacks {key}; run prepare first: {path}")
    ready_path = paths["annotation_path"].parent / "ready.json"
    if not ready_path.is_file():
        raise DatasetCatalogError("Dataset bundle is not published; run prepare first")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("schema_version") != "dataset-bundle.v2":
        raise DatasetCatalogError("Prepared version lacks current integrity contracts; prepare a new version")
    access = ready["access"]
    if ready.get("preparation_schema") is not None:
        if ready["preparation_schema"] != "preparation-archive.v1" or not access.get("preparation_path") or not access.get("preparation_digest"):
            raise DatasetCatalogError("Prepared version lacks its preparation evidence binding")
    from medbenchagent.data.integrity import verify_prepared_indexes
    try:
        verify_prepared_indexes(access)
    except ValueError as exc:
        raise DatasetCatalogError(str(exc)) from exc
    profile = json.loads(paths["profile_path"].read_text(encoding="utf-8"))
    if (Path(access["annotation_path"]).resolve() != paths["annotation_path"] or
        Path(access["profile_path"]).resolve() != paths["profile_path"] or
        not Path(access["duckdb_path"]).is_file() or
        access["annotation_digest"] != digest_file(paths["annotation_path"])):
        raise DatasetCatalogError("Prepared dataset changed; prepare a new version")
    return paths, profile, access


def validate_card_facts(card, dataset_id, paths, profile, access):
    if (card["dataset_id"] != dataset_id or card["annotation_digest"] != access["annotation_digest"] or
        card["profile_digest"] != digest_file(paths["profile_path"]) or
        card["image_manifest_digest"] != digest_file(paths["image_manifest_path"]) or
        card["total_images"] != profile["total_images"] or card["fields"] != profile["fields"] or
        card["structures"] != profile["structures"] or card["summary"] != profile["summary"]):
        raise DatasetCatalogError("Prepared dataset/Card facts changed; prepare a new version")


def load_prepared(entry):
    paths, profile, access = load_prepared_assets(entry)
    card_path = resolve_path(entry["dataset_card_path"])
    if not card_path.is_file():
        raise DatasetCatalogError(f"{entry['dataset_id']} lacks dataset_card_path; run refresh-card --dataset-id {entry['dataset_id']}")
    card = read_card(card_path)
    validate_card_facts(card.model_dump(), entry["dataset_id"], paths, profile, access)
    validate_semantics(card.semantics, profile)
    return card, access


def register_dataset(dataset_id, bundle, *, catalog_path=None):
    validate_dataset_id(dataset_id)
    bundle = Path(bundle).resolve()
    ready = json.loads((bundle / "ready.json").read_text(encoding="utf-8"))
    card_path = bundle / "dataset_card.json"
    card = read_card(card_path)
    entry = {"dataset_id": dataset_id, "name": card.name, "description": card.semantics.description,
             "available": True, "annotation_path": str(bundle / "annotation.jsonl"),
             "image_manifest_path": str(bundle / "image_manifest.jsonl"),
             "profile_path": ready["access"]["profile_path"], "dataset_card_path": str(card_path)}
    load_prepared(entry)
    target = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    entries = load_dataset_catalog(target) if target.is_file() else []
    entries = [e for e in entries if e["dataset_id"] != dataset_id] + [entry]
    atomic_write_json(target, {"datasets": sorted(entries, key=lambda e: e["dataset_id"])})
    return entry


def refresh_card(dataset_id, *, catalog_path=None):
    """Publish a new Card revision; never rebuild or overwrite prepared data or the old Card."""
    target = Path(catalog_path or settings.get_settings().paths.dataset_catalog)
    entries = load_dataset_catalog(target)
    entry = next((e for e in entries if e["dataset_id"] == dataset_id), None)
    if entry is None:
        raise DatasetCatalogError(f"Unknown dataset id: {dataset_id}")
    paths, profile, access = load_prepared_assets(entry)
    old_path = resolve_path(entry["dataset_card_path"])
    if old_path.is_file():
        # Only use the prior Card's deterministic facts to detect corrupt data. Its
        # descriptions and capability hints must never enter the migration prompt.
        previous = json.loads(old_path.read_text(encoding="utf-8"))
        facts = ("dataset_id", "annotation_digest", "profile_digest", "image_manifest_digest",
                 "total_images", "fields", "structures", "summary")
        if not all(key in previous for key in facts):
            raise DatasetCatalogError("Previous Card lacks verifiable dataset facts")
        validate_card_facts(previous, dataset_id, paths, profile, access)
    bundle = paths["annotation_path"].parent
    destination = bundle / "cards" / f"{uuid4()}.json"
    card = build_card(dataset_id, bundle, destination=destination)
    updated = {**entry, "name": card.name, "description": card.semantics.description,
               "dataset_card_path": str(destination)}
    load_prepared(updated)
    atomic_write_json(target, {"datasets": [updated if e["dataset_id"] == dataset_id else e for e in entries]})
    return updated
