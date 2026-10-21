"""Publish and export validated benchmark packages independently of agent runtime state."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

from medbenchagent.audit.task_items import _read_jsonl, _write_jsonl, audit_benchmark
from medbenchagent.evaluation.loader import PACKAGE_FILES, verify_package
from medbenchagent.data.sources import digest_file
from medbenchagent.planning.bir import BIRStore, business_dependencies, digest_file as artifact_digest
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json


def audit_candidate(context):
    """Audit and register an unpublished candidate bound to its exact input artifacts."""
    from medbenchagent.workflow.supervision import current_document
    saved = current_document(context, "candidate.json")
    if saved:
        return result("success", "Audited candidate is current", candidate_path=saved["directory"], candidate_counts=saved["counts"])
    store = BIRStore(context["bir_path"])
    dependencies = business_dependencies(context, "allocation.json",
                      "task_plan.json", "generation.json")
    identity = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
    root = Path(context["experiment_output_directory"]) / "candidates" / identity
    outcome = audit_benchmark(context, output_directory=root)
    dependencies.update({"file:" + str(path): artifact_digest(path) for path in root.iterdir() if path.is_file()})
    value = {"schema_version": "benchmark-candidate", "status": "audited",
             "directory": str(root), "counts": outcome["data"]["counts"]}
    store.update("candidate.json", value, run_id=context["run_id"], actor="audit", dependencies=dependencies)
    return result("success", outcome["summary"], candidate_path=str(root), candidate_counts=value["counts"])


def compilation_report(store, counts, limitation_reason=""):
    brief = store.load("benchmark_info.json")
    dataset = store.load("dataset_binding.json")["access"]["dataset_id"]
    plan = store.load("task_plan.json")
    lines = [f"# MedD2B compilation: {dataset}", "", "## Requirements", "",
             f"Clinical goal: {brief.get('clinical_scope') or 'Discover tasks supported by dataset evidence'}",
             f"Language: {brief.get('language')}",
             f"Requested: {counts.get('requested', 0)}; accepted: {counts.get('accepted', 0)}; shortfall: {counts.get('shortfall', 0)}",
             "", "## Tasks and evidence", ""]
    for leaf in plan.get("leaf_tasks", {}).values():
        lines.extend([f"- {leaf['name']} ({leaf['leaf_task_id']}): {leaf.get('eligible_count', 0)} eligible SamplingUnits; "
                      f"target: {leaf['target']}; reference: {leaf['reference_rule']}",
                      "  Evidence: " + json.dumps(leaf.get("evidence", []), ensure_ascii=False)])
    lines.extend(["", "## Exclusions", ""])
    for key, scenario in plan.get("scenarios", {}).items():
        if scenario.get("exclusion_reason"):
            lines.append(f"- {key} [{scenario.get('exclusion_category') or 'unsupported'}]: {scenario['exclusion_reason']}")
    lines.extend(["", "## Limitations", "", limitation_reason or "No additional delivery limitations.",
                  "", "The audit report records quarantined items and capacity shortfalls. "
                  "Model task discovery is evidence-based and is not a claim of exhaustive clinical coverage.", ""])
    return "\n".join(lines)


def publish_benchmark(context, *, limitation_reason=""):
    store = BIRStore(context["bir_path"])
    candidate = store.load("candidate.json")
    review = store.load("review.json")
    if review.get("status") != "approved" or any(row.get("blocking") for row in review.get("findings", [])):
        raise ValueError("Resolve blocking semantic review findings before publication")
    counts = candidate["counts"]
    if counts.get("accepted", 0) <= 0:
        raise ValueError("An empty candidate cannot be published")
    if (counts.get("shortfall") or counts.get("rejected")) and not limitation_reason.strip():
        raise ValueError("Publishing a limited candidate requires an explicit limitation reason")
    source = Path(candidate["directory"])
    manifest = json.loads((source / "benchmark_manifest.json").read_text(encoding="utf-8"))
    publication_id = manifest["benchmark_id"] + "-" + hashlib.sha256(limitation_reason.encode()).hexdigest()[:12]
    root = Path(context["experiment_output_directory"]) / "deliveries" / publication_id
    root.mkdir(parents=True, exist_ok=True)
    for name in PACKAGE_FILES:
        shutil.copy2(source / name, root / name)
    (root / "compilation_report.md").write_text(compilation_report(store, counts, limitation_reason), encoding="utf-8")
    files = {name: digest_file(root / name) for name in (*PACKAGE_FILES, "compilation_report.md")}
    atomic_write_json(root / "complete.json", {"schema_version": "benchmark-delivery",
                                               "benchmark_id": manifest["benchmark_id"], "files": files})
    value = {"schema_version": "bir-delivery", "status": "complete", "delivery_path": str(root),
             "manifest_path": str(root / "benchmark_manifest.json"), "combined_output": str(root / "benchmark.jsonl"),
             "counts": counts, "limitation_reason": limitation_reason}
    dependencies = business_dependencies(context, "candidate.json", "review.json")
    dependencies.update({"file:" + str(root / name): artifact_digest(root / name) for name in files})
    dependencies["file:" + str(root / "complete.json")] = artifact_digest(root / "complete.json")
    store.update("delivery.json", value, run_id=context["run_id"], actor="publication", dependencies=dependencies)
    return result("success", f"Published {counts['accepted']} benchmark items", delivery_path=str(root),
                  manifest_path=value["manifest_path"], delivery_counts=counts,
                  compilation_report_path=str(root / "compilation_report.md"))


def export_benchmark(source, destination):
    """Copy model-input assets and publish a portable package; never mutate the source."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = verify_package(source)
    export_request = {"benchmark_id": manifest["benchmark_id"], "source_digest": digest_file(source / "complete.json")}
    request_path = destination / ".export.json"
    if destination.exists():
        if (destination / "complete.json").is_file():
            completed = verify_package(destination)
            if completed.get("benchmark_id") == manifest["benchmark_id"] and completed.get("package_mode") == "portable":
                return str(destination)
        if not request_path.is_file() or json.loads(request_path.read_text(encoding="utf-8")) != export_request:
            raise ValueError("Export destination already exists and does not belong to this export")
    else:
        destination.mkdir(parents=True)
        atomic_write_json(request_path, export_request)
    asset_files = {}
    items = _read_jsonl(source / "benchmark.jsonl")
    for item in items:
        from medbenchagent.contracts import BenchmarkItem
        BenchmarkItem.model_validate(item)
        for asset in item.get("input_images", []):
            original = Path(asset["path"])
            original = original if original.is_absolute() else source / original
            digest = digest_file(original)
            if asset.get("file_digest") != digest:
                raise ValueError("Input image digest changed before export")
            relative = f"assets/{digest}{original.suffix.lower()}"
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                shutil.copy2(original, target)
            asset["path"] = relative
            asset_files[relative] = digest
    _write_jsonl(destination / "benchmark.jsonl", items)
    for name in ("audit_report.json", "quarantine.jsonl", "compilation_report.md"):
        shutil.copy2(source / name, destination / name)
    manifest.update(package_mode="portable", asset_base=".", assets=asset_files)
    atomic_write_json(destination / "benchmark_manifest.json", manifest)
    files = {name: digest_file(destination / name) for name in (*PACKAGE_FILES, "compilation_report.md")}
    atomic_write_json(destination / "complete.json", {"schema_version": "benchmark-delivery",
        "benchmark_id": manifest["benchmark_id"], "files": {**files, **asset_files}})
    verify_package(destination)
    request_path.unlink(missing_ok=True)
    return str(destination)
