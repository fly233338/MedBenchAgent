"""Deterministic portable publication, independent of BIR, old workflows and model runtimes."""

import hashlib
import json
from pathlib import Path

from medbenchagent.audit.generation import audit_candidate
from medbenchagent.data.dataset import file_digest
from medbenchagent.data.task_data import write_once
from medbenchagent.evaluation.loader import load_benchmark_package


def publish_delivery(
    destination,
    *,
    candidate,
    candidate_binding,
    audit,
    audit_binding,
    review,
    review_binding,
    executions,
    limitation_reason="",
):
    candidate.verify_integrity()
    if (
        review.target != candidate_binding
        or audit.target != candidate_binding
        or review.audit != audit_binding
        or not review.approved
        or any(f.blocking for f in review.findings)
        or review.unchecked_scope
        or not audit.passed
    ):
        raise ValueError(
            "Delivery requires matching approved Review and passing Audit without blocking findings"
        )
    current_audit = audit_candidate(candidate, candidate_binding, executions)
    if current_audit != audit:
        raise ValueError("Audit no longer matches actual candidate contents")

    if not candidate.items:
        raise ValueError("An empty candidate cannot be delivered")
    counts = candidate.manifest.counts
    if (counts["shortfall"] or counts["quarantined"]) and not limitation_reason.strip():
        raise ValueError("Limited delivery requires an explicit limitation reason")
    task_ids = set(candidate.manifest.allocation.counts)
    item_map = {item.item_id: item for item in candidate.items}
    if (
        set(review.checked_task_ids) != task_ids
        or set(review.checked_item_ids) - item_map.keys()
        or {item.leaf_task_id for item in candidate.items}
        - {item_map[key].leaf_task_id for key in review.checked_item_ids}
    ):
        raise ValueError("Semantic review coverage does not match the candidate")
    root = Path(destination).resolve()
    assets, items = {}, []
    for item in candidate.items:
        value = item.model_dump(mode="json")
        for entry in value["inputs"]:
            if entry["kind"] != "asset":
                continue
            asset = entry["value"]
            original = Path(asset["path"])
            digest = file_digest(original)
            if digest != asset["file_digest"]:
                raise ValueError("Input resource changed before delivery")
            relative = f"assets/{digest}{original.suffix.lower()}"
            write_once(root / relative, original.read_bytes())
            asset["path"] = relative
            assets[relative] = digest
        items.append(value)
    dependencies = {
        "candidate": candidate_binding.model_dump(),
        "audit": audit_binding.model_dump(),
        "review": review_binding.model_dump(),
    }
    identity = hashlib.sha256(
        json.dumps(
            {"dependencies": dependencies, "limitation": limitation_reason},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    tasks, plans, specs = {}, {}, {}
    for spec_ref, spec, plan, _ in executions.values():
        task = plan.tasks[spec.task_id]
        tasks[f"{task.dataset.dataset_id}::{task.task_id}"] = {
            "dataset_id": task.dataset.dataset_id,
            "leaf_task_id": task.task_id,
            "scenario_id": task.scenario_id,
            "task_family": task.task_family,
            "capability_category": task.capability_category,
            "output_spec": task.assembly.output_spec,
        }
        plans[spec.task_plan.artifact_id] = plan.model_dump(mode="json")
        specs[spec_ref.artifact_id] = spec.model_dump(mode="json")
    manifest = {
        "schema_version": "benchmark-manifest",
        "benchmark_id": identity,
        "dependencies": dependencies,
        "package_mode": "portable",
        "asset_base": ".",
        "assets": assets,
        "tasks": tasks,
        "item_count": len(items),
        "counts": counts,
        "limitation_reason": limitation_reason,
    }
    files = {
        "benchmark.jsonl": "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        "quarantine.jsonl": "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in candidate.quarantine
        ),
        "benchmark_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2),
        "audit_report.json": audit.model_dump_json(indent=2),
        "review_report.json": review.model_dump_json(indent=2),
        "task_plans.json": json.dumps(plans, ensure_ascii=False, indent=2),
        "generation_specs.json": json.dumps(specs, ensure_ascii=False, indent=2),
        "compilation_report.md": f"# Benchmark compilation\n\nRequested: {counts['requested']}; accepted: {counts['accepted']}; "
        f"shortfall: {counts['shortfall']}; quarantined: {counts['quarantined']}.\n\n"
        f"Semantic review sampled: {review.semantic_sampled}.\n\nLimitations: {limitation_reason or 'None recorded'}.\n",
    }
    for name, content in files.items():
        write_once(root / name, content.encode("utf-8"))
    marker = {
        "schema_version": "benchmark-delivery",
        "benchmark_id": identity,
        "counts": counts,
        "files": {**assets, **{name: file_digest(root / name) for name in files}},
    }
    write_once(
        root / "complete.json",
        (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode(),
    )
    load_benchmark_package(root)
    return root / "complete.json"


def export_benchmark(source, destination):
    """Copy exactly the files bound by a portable package, preserving review evidence."""
    from medbenchagent.evaluation.loader import verify_package

    source, destination = Path(source).resolve(), Path(destination).resolve()
    verify_package(source)
    load_benchmark_package(source)
    if destination.exists():
        raise ValueError("Export destination already exists")
    marker = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    for name in [*marker["files"], "complete.json"]:
        write_once(destination / name, (source / name).read_bytes())
    load_benchmark_package(destination)
    return destination
