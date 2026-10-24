"""Register source or current canonical Dataset manifests, without legacy bundles."""

import json
import re
from pathlib import Path

from filelock import FileLock

from medbenchagent.settings import load_paths
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.data.task_data import SourceManifest


def catalog_path(catalog=None):
    return (
        Path(catalog)
        if catalog is not None
        else load_paths()["runtime_root"].parent / "datasets.json"
    )


def validate_dataset_id(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError("Invalid dataset ID")
    return value


def list_datasets(catalog=None):
    path = catalog_path(catalog)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def register_dataset(dataset_id, manifest, *, kind="dataset", catalog=None):
    validate_dataset_id(dataset_id)
    path = Path(manifest).resolve()
    if kind == "dataset":
        dataset = FileDatasetReader(path).get_manifest()
    elif kind == "source":
        dataset = SourceManifest.model_validate_json(path.read_bytes()).dataset
    else:
        raise ValueError("Expected source or dataset")
    if dataset.dataset_id != dataset_id:
        raise ValueError("Registration ID differs from the manifest Dataset")
    catalog = catalog_path(catalog)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(catalog) + ".lock"):
        entries = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else {}
        entry = {"kind": kind, "manifest": str(path), "revision": dataset.revision}
        existing = entries.get(dataset_id)
        if existing and existing["revision"] == entry["revision"] and existing["kind"] != kind:
            raise ValueError("A Dataset revision cannot change its input kind")
        entries[dataset_id] = entry
        atomic_write_json(catalog, entries)
    return entry


def remove_registration(dataset_id, catalog=None):
    """Remove only the reusable index; never delete source or Run data."""
    catalog = catalog_path(catalog)
    with FileLock(str(catalog) + ".lock"):
        entries = list_datasets(catalog)
        entries.pop(dataset_id, None)
        atomic_write_json(catalog, entries)


def registered_input(dataset_id, catalog=None):
    entry = list_datasets(catalog)[dataset_id]
    return {
        "source_manifest" if entry["kind"] == "source" else "dataset_manifest": Path(
            entry["manifest"]
        )
    }
