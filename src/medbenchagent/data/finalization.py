"""Deterministic publication of annotation.v5 bundles."""

from pathlib import Path
import json
import shutil
from medbenchagent.data.v5_validation import validate_v5_candidate, write_v5_report
from medbenchagent.data.v5_index import prepare_v5_dataset
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json


def finalize_v5_prepared_dataset(candidate, context):
    candidate = Path(candidate).resolve()
    report = validate_v5_candidate(
        candidate,
        source_root=context.get("source_root"),
        source_roots=context.get("source_roots"),
    )
    write_v5_report(candidate, report)
    if report["systemic_failure"] or not report["accepted_cases"]:
        raise ValueError("v5 candidate cannot be published")
    bundle = Path(context["bundle_root"]).resolve()
    temporary = bundle.with_name(bundle.name + ".publishing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    names = (
        "cases.jsonl",
        "asset_manifest.jsonl",
        "provenance.jsonl",
        "preparation_report.json",
        "quarantine.jsonl",
        "optional_evidence_dropped.jsonl",
        "conversion_config.json",
        "grouping_policy.json",
        "mapping.json",
        "conversion_version",
        "complete.json",
    )
    for name in names:
        shutil.copy2(candidate / name, temporary / name)
    for relative in ("scripts/prepare.py", "notes/inventory.md", "notes/mapping.md", "notes/dataset_notes.md"):
        source = candidate / relative
        if source.is_file():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    profile, access = prepare_v5_dataset(
        temporary / "cases.jsonl",
        temporary / "asset_manifest.jsonl",
        temporary / "cache",
        revision="1",
        source_root=context.get("source_root"),
        source_roots=context.get("source_roots"),
    )
    atomic_write_json(temporary / "dataset_profile.json", profile)
    dataset_id = context.get("dataset_id") or context.get("dataset_scope") or "dataset"
    card = {
        "schema_version": "dataset-card.v3",
        "dataset_id": dataset_id,
        "name": dataset_id,
        "description": "annotation.v5 dataset",
        "annotation_digest": digest_file(temporary / "cases.jsonl"),
        "asset_manifest_digest": digest_file(temporary / "asset_manifest.jsonl"),
        "profile_digest": digest_file(temporary / "dataset_profile.json"),
        "case_count": profile["case_count"],
        "patient_count": profile["patient_count"],
        "evidence_profile": profile["evidence"],
        "asset_profile": profile["assets"],
    }
    atomic_write_json(temporary / "dataset_card.json", card)
    (temporary / "presentation_manifest.jsonl").write_text("", encoding="utf-8")
    access = {
        **access,
        "cases_path": str(temporary / "cases.jsonl"),
        "asset_manifest_path": str(temporary / "asset_manifest.jsonl"),
        "profile_path": str(temporary / "dataset_profile.json"),
        "dataset_card_path": str(temporary / "dataset_card.json"),
        "presentation_manifest_path": str(temporary / "presentation_manifest.jsonl"),
        "duckdb_path": str(temporary / "cache" / "annotation-cache.v5" / "1" / "dataset.duckdb"),
    }
    atomic_write_json(
        temporary / "ready.json",
        {
            "schema_version": "dataset-bundle.v4",
            "annotation_schema": "annotation.v5",
            "access": access,
        },
    )
    if bundle.exists():
        shutil.rmtree(bundle)
    temporary.replace(bundle)
    ready = json.loads((bundle / "ready.json").read_text(encoding="utf-8"))
    ready["access"].update(
        {
            "cases_path": str(bundle / "cases.jsonl"),
            "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
            "profile_path": str(bundle / "dataset_profile.json"),
            "dataset_card_path": str(bundle / "dataset_card.json"),
            "duckdb_path": str(bundle / "cache" / "annotation-cache.v5" / "1" / "dataset.duckdb"),
        }
    )
    atomic_write_json(bundle / "ready.json", ready)
    return {
        "status": "complete",
        "bundle": str(bundle),
        "cases_path": str(bundle / "cases.jsonl"),
        "access_manifest": {
            **access,
            "cases_path": str(bundle / "cases.jsonl"),
            "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
            "profile_path": str(bundle / "dataset_profile.json"),
            "dataset_card_path": str(bundle / "dataset_card.json"),
            "duckdb_path": str(bundle / "cache" / "annotation-cache.v5" / "1" / "dataset.duckdb"),
        },
        "report": report,
        "card_path": str(bundle / "dataset_card.json"),
        "dataset_entry": {
            "dataset_id": dataset_id,
            "name": dataset_id,
            "description": "annotation.v5 dataset",
            "available": True,
            "cases_path": str(bundle / "cases.jsonl"),
            "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
            "profile_path": str(bundle / "dataset_profile.json"),
            "dataset_card_path": str(bundle / "dataset_card.json"),
        },
    }


def finalize_prepared_dataset(candidate, context):
    return finalize_v5_prepared_dataset(candidate, context)
