"""Deterministic publication of validated preparation candidates."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from medbenchagent.data.preparation_models import RegisteredDatasetRef, ValidatedCandidateRef, ArtifactRef
from medbenchagent.runtime.run_artifacts import atomic_write_json


REQUIRED = ("annotation.jsonl", "annotation_fields.json", "image_manifest.jsonl",
            "provenance.jsonl", "transforms.jsonl", "reconciliation.json", "quarantine.jsonl")


def publish_validated_candidate(candidate: ValidatedCandidateRef, *, dataset_id: str,
                                destination: Path, catalog_path: Path) -> RegisteredDatasetRef:
    source = Path(candidate.root).resolve()
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing or not (source / "images").is_dir():
        raise ValueError(f"Validated candidate is incomplete: {', '.join(missing)}")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = destination / dataset_id
    ready = target / "ready.json"
    if not ready.exists():
        staging = destination / f".{dataset_id}.{uuid4().hex}.staging"
        shutil.copytree(source, staging)
        access = {name: str(staging / name) for name in REQUIRED}
        access["bundle_root"] = str(staging)
        atomic_write_json(staging / "ready.json", {"schema_version": "dataset-bundle.v3", "access": access,
                                                     "validation": candidate.validation_ref.model_dump()})
        staging.replace(target)
    payload = json.loads(ready.read_text(encoding="utf-8"))
    entry = {"dataset_id": dataset_id, "available": True, "bundle_root": str(target),
             "annotation_path": str(target / "annotation.jsonl"),
             "image_manifest_path": str(target / "image_manifest.jsonl"),
             "dataset_access_manifest": str(target / "ready.json")}
    catalog_path = Path(catalog_path).resolve()
    with FileLock(str(catalog_path) + ".lock"):
        catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.is_file() else {"datasets": []}
        datasets = [item for item in catalog.get("datasets", []) if item.get("dataset_id") != dataset_id]
        datasets.append(entry)
        atomic_write_json(catalog_path, {"datasets": sorted(datasets, key=lambda item: item["dataset_id"])})
    return RegisteredDatasetRef(dataset_id=dataset_id, bundle_root=str(target), version="3",
                                dataset_entry=entry, annotation_path=entry["annotation_path"],
                                dataset_access_manifest=entry["dataset_access_manifest"],
                                validation_ref=ArtifactRef.model_validate(payload["validation"]))
