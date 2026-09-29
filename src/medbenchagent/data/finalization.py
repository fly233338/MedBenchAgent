"""Deterministic finalization of a Coding Agent dataset candidate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from medbenchagent.data.dataset import prepare_annotation_dataset
from medbenchagent.data.dataset_card import build_card
from medbenchagent.data.dataset_catalog import register_dataset
from medbenchagent.data.validation import iter_annotation_jsonl, validate_v4_record, validate_dataset_files
from medbenchagent.runtime.run_artifacts import atomic_write_json


def finalize_prepared_dataset(candidate, context):
    """Validate, materialize, index, card and register a submitted candidate.

    The candidate is never edited in place. A fresh bundle is created for every
    publication attempt, which makes retries and recovery idempotent.
    """
    candidate = Path(candidate).resolve()
    required = ("annotation.jsonl", "annotation_fields.json", "image_manifest.jsonl", "provenance.jsonl")
    missing = [name for name in required if not (candidate / name).is_file()]
    report_dir = Path(context.get("operation_directory", candidate)) / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    if missing:
        report = {"schema_version": "prepared-validation.v1", "status": "failed", "missing": missing}
        atomic_write_json(report_dir / "report.json", report)
        raise ValueError("Candidate is missing required outputs: " + ", ".join(missing))

    count = 0
    try:
        with (candidate / "annotation.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                validate_v4_record(json.loads(line), line_number)
                count += 1
        if not count:
            raise ValueError("Candidate annotation is empty")
        # Keep the existing streaming checks for image identity and duplicate paths.
        list(iter_annotation_jsonl(candidate / "annotation.jsonl"))
        # Dimensions and population are checked against the candidate manifest before
        # anything is copied into the publication bundle.
        validate_dataset_files(candidate / "annotation.jsonl", candidate / "image_manifest.jsonl")
        with (candidate / "provenance.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                item = json.loads(line)
                if not isinstance(item, dict) or not item.get("source_id"):
                    raise ValueError(f"Invalid provenance at line {line_number}")
    except Exception as exc:
        atomic_write_json(report_dir / "report.json", {
            "schema_version": "prepared-validation.v1", "status": "failed", "error": str(exc)
        })
        raise

    bundle = Path(context["bundle_root"]).resolve()
    if bundle.exists() and any(bundle.iterdir()):
        raise ValueError("Publication bundle is not empty; use a new dataset version")
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate / "annotation.jsonl", bundle / "annotation.jsonl")
    shutil.copy2(candidate / "annotation_fields.json", bundle / "annotation_fields.json")
    shutil.copy2(candidate / "provenance.jsonl", bundle / "provenance.jsonl")
    shutil.copy2(candidate / "image_manifest.jsonl", bundle / "image_manifest.jsonl")
    if (candidate / "images").is_dir():
        shutil.copytree(candidate / "images", bundle / "images", dirs_exist_ok=True)
    profile, access = prepare_annotation_dataset(bundle / "annotation.jsonl", bundle / "cache",
                                                 scratch_root=report_dir / "profile")
    access.update(annotation_schema="annotation.v4",
                  annotation_path=str(bundle / "annotation.jsonl"),
                  image_manifest_path=str(bundle / "image_manifest.jsonl"))
    atomic_write_json(bundle / "ready.json", {"schema_version": "dataset-bundle.v2", "access": access,
                                               "sources": sorted(context.get("sources", []), key=str)})
    build_card(context["dataset_id"], bundle)
    entry = register_dataset(context["dataset_id"], bundle,
                             catalog_path=context.get("registration_catalog_path"))
    report = {"schema_version": "prepared-validation.v1", "status": "complete", "records": count,
              "bundle": str(bundle), "dataset_entry": entry}
    atomic_write_json(report_dir / "report.json", report)
    return {"status": "complete", "dataset_entry": entry, "annotation_path": str(bundle / "annotation.jsonl"),
            "card_path": entry["dataset_card_path"], "access_manifest": access,
            "validation_report": str(report_dir / "report.json")}
