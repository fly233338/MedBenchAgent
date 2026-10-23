"""Register source or current canonical Dataset manifests, without legacy bundles."""
import json
import re
from pathlib import Path
from filelock import FileLock
from medbenchagent.settings import PROJECT_ROOT
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.data.task_data import SourceManifest

CATALOG_PATH = PROJECT_ROOT / ".runtime" / "datasets.json"


def validate_dataset_id(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError("Invalid dataset ID")
    return value


def register_dataset(dataset_id, manifest, *, kind="dataset", catalog=CATALOG_PATH):
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
    catalog = Path(catalog)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(catalog) + ".lock"):
        entries = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else {}
        entry = {"kind": kind, "manifest": str(path)}
        if dataset_id in entries and entries[dataset_id] != entry:
            raise ValueError("Dataset registration already points to another manifest")
        entries[dataset_id] = entry
        atomic_write_json(catalog, entries)
    return entry


def registered_input(dataset_id, catalog=CATALOG_PATH):
    entry = json.loads(Path(catalog).read_text(encoding="utf-8"))[dataset_id]
    return {"source_manifest" if entry["kind"] == "source" else "dataset_manifest": Path(entry["manifest"])}
