"""Deterministic publication of annotation bundles."""

from pathlib import Path
import json
import shutil
from medbenchagent.data.preparation_validation import validate_candidate, write_validation_report
from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest, load_manifest


def finalize_prepared_dataset(candidate, context):
    candidate = Path(candidate).resolve()
    report = validate_candidate(
        candidate,
        source_root=context.get("source_root"),
        source_roots=context.get("source_roots"),
    )
    write_validation_report(candidate, report)
    if report["systemic_failure"] or not report["accepted_cases"]:
        raise ValueError("Prepared candidate cannot be published")
    atomic_write_json(candidate / "complete.json", {
        "status": "complete",
        "accepted_cases": report["accepted_cases"],
        "quarantined_cases": report["quarantined_cases"],
        "preparation_report": str((candidate / "preparation_report.json").resolve()),
    })
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
    optional_manifests = {
        "presentation_manifest.jsonl": PresentationRecord,
        "split_manifest.jsonl": SplitManifest,
        "sampling_manifest.jsonl": SamplingManifest,
    }
    for name, model in optional_manifests.items():
        source = candidate / name
        if not source.is_file():
            continue
        parsed = load_manifest(source, model)
        if name == "presentation_manifest.jsonl":
            records = parsed if isinstance(parsed, list) else [parsed]
            normalized_records = []
            for record in records:
                source_asset = Path(record.path)
                if not source_asset.is_absolute():
                    source_asset = candidate / source_asset
                source_asset = source_asset.resolve()
                relative = source_asset.relative_to(candidate)
                target_asset = temporary / relative
                target_asset.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_asset, target_asset)
                normalized_records.append(record.model_copy(update={"path": relative.as_posix()}))
            with (temporary / name).open("w", encoding="utf-8", newline="\n") as handle:
                for record in normalized_records:
                    handle.write(record.model_dump_json() + "\n")
        else:
            shutil.copy2(source, temporary / name)
    profile, access = prepare_dataset(
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
        "schema_version": "dataset-card",
        "dataset_id": dataset_id,
        "name": dataset_id,
        "description": "annotation dataset",
        "annotation_digest": digest_file(temporary / "cases.jsonl"),
        "asset_manifest_digest": digest_file(temporary / "asset_manifest.jsonl"),
        "profile_digest": digest_file(temporary / "dataset_profile.json"),
        "case_count": profile["case_count"],
        "patient_count": profile["patient_count"],
        "evidence_profile": profile["evidence"],
        "asset_profile": profile["assets"],
    }
    atomic_write_json(temporary / "dataset_card.json", card)
    access = {
        **access,
        "cases_path": str(temporary / "cases.jsonl"),
        "asset_manifest_path": str(temporary / "asset_manifest.jsonl"),
        "profile_path": str(temporary / "dataset_profile.json"),
        "mapping_path": str(temporary / "mapping.json"),
        "dataset_card_path": str(temporary / "dataset_card.json"),
        "duckdb_path": str(temporary / "cache" / "annotation-cache" / "1" / "dataset.duckdb"),
        "source_root": context.get("source_root"),
        "source_roots": context.get("source_roots", {}),
    }
    atomic_write_json(
        temporary / "ready.json",
        {
            "schema_version": "dataset-bundle",
            "annotation_schema": "annotation",
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
            "mapping_path": str(bundle / "mapping.json"),
            "dataset_card_path": str(bundle / "dataset_card.json"),
            "duckdb_path": str(bundle / "cache" / "annotation-cache" / "1" / "dataset.duckdb"),
        }
    )
    for name in ("presentation", "split", "sampling"):
        key = f"{name}_manifest_path"
        if key in ready["access"]:
            ready["access"][key] = str(bundle / f"{name}_manifest.jsonl")
    atomic_write_json(bundle / "ready.json", ready)
    access_manifest = {
        **access,
        "cases_path": str(bundle / "cases.jsonl"),
        "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
        "profile_path": str(bundle / "dataset_profile.json"),
        "mapping_path": str(bundle / "mapping.json"),
        "dataset_card_path": str(bundle / "dataset_card.json"),
        "duckdb_path": str(bundle / "cache" / "annotation-cache" / "1" / "dataset.duckdb"),
    }
    for name in ("presentation", "split", "sampling"):
        key = f"{name}_manifest_path"
        if key in access_manifest:
            access_manifest[key] = str(bundle / f"{name}_manifest.jsonl")
    return {
        "status": "complete",
        "bundle": str(bundle),
        "cases_path": str(bundle / "cases.jsonl"),
        "access_manifest": access_manifest,
        "report": report,
        "card_path": str(bundle / "dataset_card.json"),
        "dataset_entry": {
            "dataset_id": dataset_id,
            "name": dataset_id,
            "description": "annotation dataset",
            "available": True,
            "cases_path": str(bundle / "cases.jsonl"),
            "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
            "profile_path": str(bundle / "dataset_profile.json"),
            "dataset_card_path": str(bundle / "dataset_card.json"),
        },
    }

