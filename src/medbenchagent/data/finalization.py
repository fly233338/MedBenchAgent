"""Deterministic finalization of a Coding Agent dataset candidate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from medbenchagent.data.dataset import prepare_annotation_dataset
from medbenchagent.data.dataset_card import build_card
from medbenchagent.data.dataset_catalog import register_dataset
from medbenchagent.data.validation import (
    iter_annotation_jsonl, validate_v4_record, validate_dataset_files,
    validate_candidate_sidecars,
)
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.data.sources import digest_file


def finalize_v5_prepared_dataset(candidate, context):
    """Publish a canonical v5 preparation bundle without touching legacy paths."""
    from medbenchagent.data.v5_validation import validate_v5_candidate, write_v5_report

    candidate = Path(candidate).resolve()
    report = validate_v5_candidate(candidate, source_root=context.get("source_root"))
    write_v5_report(candidate, report)
    if report["systemic_failure"] or not report["accepted_cases"]:
        raise ValueError("v5 candidate cannot be published")
    bundle = Path(context["bundle_root"]).resolve()
    if (bundle / "ready.json").is_file():
        raise ValueError("Publication bundle is already complete; use a new dataset version")
    temporary = bundle.with_name(bundle.name + ".publishing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    for name in ("annotation.jsonl", "asset_manifest.jsonl", "provenance.jsonl", "preparation_report.json", "quarantine.jsonl", "optional_evidence_dropped.jsonl"):
        shutil.copy2(candidate / name, temporary / name)
    atomic_write_json(temporary / "conversion_config.json", {"schema_version": "conversion-config.v5", "converter_version": "v5"})
    atomic_write_json(temporary / "grouping_policy.json", {"schema_version": "grouping-policy.v5", "priority": ["study_or_exam", "documented_case", "documented_composite", "record_or_asset_bundle"], "patient_id_never_merges": True})
    atomic_write_json(temporary / "mapping.json", {"schema_version": "mapping.v5", "evidence_types": ["value", "text", "asset", "region"]})
    (temporary / "conversion_version").write_text("v5\n", encoding="utf-8")
    (temporary / "presentation_manifest.jsonl").write_text("", encoding="utf-8")
    from medbenchagent.data.dataset import prepare_annotation_dataset
    profile, indexed_access = prepare_annotation_dataset(temporary / "annotation.jsonl", temporary / "cache", revision="1")
    atomic_write_json(temporary / "dataset_profile.json", profile)
    dataset_id = context.get("dataset_id") or context.get("dataset_scope") or "dataset"
    card = {
        "schema_version": "dataset-card.v3", "dataset_id": dataset_id,
        "name": dataset_id, "description": "annotation.v5 dataset",
        "annotation_digest": digest_file(temporary / "annotation.jsonl"),
        "asset_manifest_digest": digest_file(temporary / "asset_manifest.jsonl"),
        "profile_digest": digest_file(temporary / "dataset_profile.json"),
        "case_count": profile["case_count"], "patient_count": profile["patient_count"],
        "evidence_profile": profile["evidence"], "asset_profile": profile["assets"],
        "summary": {"case_count": profile["case_count"], "patient_count": profile["patient_count"]},
    }
    atomic_write_json(temporary / "dataset_card.json", card)
    access = {
        "schema_version": "annotation.v5",
        "annotation_path": str(temporary / "annotation.jsonl"),
        "asset_manifest_path": str(temporary / "asset_manifest.jsonl"),
        "profile_path": str(temporary / "dataset_profile.json"),
        "annotation_digest": digest_file(temporary / "annotation.jsonl"),
        "asset_manifest_digest": digest_file(temporary / "asset_manifest.jsonl"),
        "profile_digest": digest_file(temporary / "dataset_profile.json"),
        "dataset_card_path": str(temporary / "dataset_card.json"),
        "presentation_manifest_path": str(temporary / "presentation_manifest.jsonl"),
        "duckdb_path": indexed_access["duckdb_path"],
        "duckdb_digest": indexed_access["duckdb_digest"],
    }
    atomic_write_json(temporary / "ready.json", {"schema_version": "dataset-bundle.v4", "annotation_schema": "annotation.v5", "access": access})
    if bundle.exists():
        shutil.rmtree(bundle)
    temporary.replace(bundle)
    ready_path = bundle / "ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["access"].update({
        "annotation_path": str(bundle / "annotation.jsonl"),
        "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
        "profile_path": str(bundle / "dataset_profile.json"),
        "dataset_card_path": str(bundle / "dataset_card.json"),
        "duckdb_path": str(bundle / "cache" / "annotation-cache.v5" / "1" / "dataset.duckdb"),
    })
    ready_path.write_text(json.dumps(ready, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    entry = {"dataset_id": dataset_id, "name": dataset_id, "description": card["description"], "available": True,
             "annotation_path": str(bundle / "annotation.jsonl"), "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
             "profile_path": str(bundle / "dataset_profile.json"), "dataset_card_path": str(bundle / "dataset_card.json")}
    catalog_path = context.get("registration_catalog_path")
    if catalog_path:
        target = Path(catalog_path)
        catalog = {"schema_version": "dataset-catalog.v5", "datasets": []}
        if target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                if existing.get("schema_version") == "dataset-catalog.v5":
                    catalog = existing
            except json.JSONDecodeError:
                pass
        catalog["datasets"] = [item for item in catalog["datasets"] if item.get("dataset_id") != dataset_id] + [entry]
        atomic_write_json(target, catalog)
    return {"status": "complete", "bundle": str(bundle), "dataset_entry": entry, "access_manifest": {
        **access,
        "annotation_path": str(bundle / "annotation.jsonl"),
        "asset_manifest_path": str(bundle / "asset_manifest.jsonl"),
        "profile_path": str(bundle / "dataset_profile.json"),
        "dataset_card_path": str(bundle / "dataset_card.json"),
    }, "report": report, "card_path": str(bundle / "dataset_card.json")}


def finalize_prepared_dataset(candidate, context):
    """Validate, materialize, index, card and register a submitted candidate.

    The candidate is never edited in place. A fresh bundle is created for every
    publication attempt, which makes retries and recovery idempotent.
    """
    candidate = Path(candidate).resolve()
    if (candidate / "asset_manifest.jsonl").is_file():
        result = finalize_v5_prepared_dataset(candidate, context)
        dataset_id = context.get("dataset_id") or context.get("dataset_scope")
        access = result["access_manifest"]
        entry = result.get("dataset_entry") or {"dataset_id": dataset_id, "name": dataset_id or "dataset",
                 "description": "annotation.v5 dataset", "available": True,
                 "annotation_path": access["annotation_path"], "asset_manifest_path": access["asset_manifest_path"],
                 "profile_path": access["profile_path"]}
        return {"status": "complete", "dataset_id": dataset_id, "dataset_entry": entry,
                "annotation_path": access["annotation_path"], "access_manifest": access,
                "validation_report": str(Path(result["bundle"]) / "preparation_report.json"),
                "card_path": result.get("card_path"), "annotation_schema": "annotation.v5"}
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
        validate_candidate_sidecars(candidate)
        # Dimensions and population are checked against the candidate manifest before
        # anything is copied into the publication bundle.
        source_roots = {item.get("id"): item.get("root") for item in context.get("sources", []) if isinstance(item, dict) and item.get("root")}
        validate_dataset_files(candidate / "annotation.jsonl", candidate / "image_manifest.jsonl", source_roots=source_roots)
        with (candidate / "provenance.jsonl").open(encoding="utf-8") as handle:
            allowed_sources = {item.get("id") for item in context.get("sources", []) if isinstance(item, dict)}
            for line_number, line in enumerate(handle, 1):
                item = json.loads(line)
                if not isinstance(item, dict) or not item.get("source_id"):
                    raise ValueError(f"Invalid provenance at line {line_number}")
                if allowed_sources and item["source_id"] not in allowed_sources:
                    raise ValueError(f"Provenance source is not selected: {item['source_id']}")
    except Exception as exc:
        atomic_write_json(report_dir / "report.json", {
            "schema_version": "prepared-validation.v1", "status": "failed", "error": str(exc)
        })
        raise

    bundle = Path(context["bundle_root"]).resolve()
    if bundle.exists() and (bundle / "ready.json").is_file():
        raise ValueError("Publication bundle is already complete; use a new dataset version")
    temporary = bundle.with_name(bundle.name + ".publishing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    if bundle.is_dir() and (bundle / "execution_runtime.json").is_file():
        shutil.copy2(bundle / "execution_runtime.json", temporary / "execution_runtime.json")
    shutil.copy2(candidate / "annotation.jsonl", temporary / "annotation.jsonl")
    shutil.copy2(candidate / "annotation_fields.json", temporary / "annotation_fields.json")
    shutil.copy2(candidate / "provenance.jsonl", temporary / "provenance.jsonl")
    shutil.copy2(candidate / "image_manifest.jsonl", temporary / "image_manifest.jsonl")
    source_roots = {item.get("id"): item.get("root") for item in context.get("sources", []) if isinstance(item, dict) and item.get("root")}
    integrity_path = temporary / "asset_integrity.jsonl"
    with (candidate / "image_manifest.jsonl").open(encoding="utf-8") as manifest, integrity_path.open("w", encoding="utf-8") as integrity:
        for line_number, line in enumerate(manifest, 1):
            item = json.loads(line)
            relative = item.get("source_path") or item.get("png_path") or item.get("image_path")
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f"Invalid image path at manifest line {line_number}")
            if item.get("source_id"):
                from medbenchagent.data.image_assets import resolve_source_image
                asset = resolve_source_image(item["source_id"], relative, source_roots, item.get("file_digest"))
                integrity.write(json.dumps({"image_id": item.get("image_id"), "source_id": item["source_id"],
                                            "source_path": relative, "file_digest": digest_file(asset)}, ensure_ascii=False) + "\n")
            else:
                asset = (temporary / relative).resolve()
                if not asset.is_file() or not asset.is_relative_to(temporary):
                    raise ValueError(f"Published image is missing: {relative}")
                integrity.write(json.dumps({"image_id": item.get("image_id"), "png_path": relative,
                                            "file_digest": digest_file(asset)}, ensure_ascii=False) + "\n")
    documentation = temporary / "documentation" / "index.json"
    atomic_write_json(documentation, {"schema_version": "dataset-documentation.v1", "documents": []})
    profile, access = prepare_annotation_dataset(temporary / "annotation.jsonl", temporary / "cache",
                                                 scratch_root=report_dir / "profile")
    access.update(annotation_schema="annotation.v4",
                  annotation_path=str(temporary / "annotation.jsonl"),
                  image_manifest_path=str(temporary / "image_manifest.jsonl"),
                  image_integrity_path=str(integrity_path),
                  image_integrity_digest=digest_file(integrity_path),
                  documentation_path=str(documentation),
                  documentation_digest=digest_file(documentation),
                  source_roots=source_roots)
    atomic_write_json(temporary / "ready.json", {"schema_version": "dataset-bundle.v3", "annotation_schema": "annotation.v4",
                                               "access": access,
                                               "sources": sorted(context.get("sources", []), key=str)})
    build_card(context["dataset_id"], temporary)
    access = {key: (value.replace(str(temporary), str(bundle)) if isinstance(value, str) else value)
              for key, value in access.items()}
    atomic_write_json(temporary / "ready.json", {"schema_version": "dataset-bundle.v3", "annotation_schema": "annotation.v4",
                                               "access": access,
                                               "sources": sorted(context.get("sources", []), key=str)})
    if bundle.exists():
        shutil.rmtree(bundle)
    temporary.replace(bundle)
    entry = register_dataset(context["dataset_id"], bundle,
                             catalog_path=context.get("registration_catalog_path"))
    report = {"schema_version": "prepared-validation.v1", "status": "complete", "records": count,
              "bundle": str(bundle), "dataset_entry": entry}
    atomic_write_json(report_dir / "report.json", report)
    access = {**access, "annotation_path": str(bundle / "annotation.jsonl"),
              "image_manifest_path": str(bundle / "image_manifest.jsonl")}
    return {"status": "complete", "dataset_id": context["dataset_id"], "dataset_entry": entry,
            "annotation_path": str(bundle / "annotation.jsonl"),
            "card_path": entry["dataset_card_path"], "access_manifest": access,
            "validation_report": str(report_dir / "report.json")}
