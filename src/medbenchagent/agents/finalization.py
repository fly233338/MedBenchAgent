"""Deterministic audit of one dataset's generated candidate."""

from __future__ import annotations
from medbenchagent.planning.bir import BIRStore

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.audit.task_items import audit_task_items
from medbenchagent.data.manifests import SamplingManifest, SplitManifest, load_manifest, verify_manifest_digest
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json




def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"__invalid__": True, "line": line_number, "error": str(exc)})
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return str(destination)


def _option_distribution(items: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, Counter] = defaultdict(Counter)
    for item in items:
        options = item.get("options")
        if isinstance(options, list) and item.get("answer") in options:
            distribution[str(item.get("task_id"))][str(options.index(item["answer"]))] += 1
    return {task: dict(sorted(counts.items())) for task, counts in sorted(distribution.items())}


def balance_option_positions(items: list[dict[str, Any]], seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance answer positions while preserving semantic answer values."""
    before = _option_distribution(items)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if isinstance(item.get("options"), list) and item.get("options") and item.get("answer") in item["options"]:
            groups[str(item.get("task_id"))].append(item)
    for task_id, group in groups.items():
        group.sort(key=lambda item: hashlib.sha256(
            json.dumps([seed, task_id, item.get("item_id", item.get("id"))], separators=(",", ":")).encode()
        ).hexdigest())
        for rank, item in enumerate(group):
            options = list(item["options"])
            answer = item["answer"]
            target = rank % len(options)
            current = options.index(answer)
            shift = (target - current) % len(options)
            item["options"] = options[-shift:] + options[:-shift] if shift else options
    return items, {"before": before, "after": _option_distribution(items)}


def _limit_reasons(items: Iterable[dict[str, Any]], *, max_tasks_per_image: int | None,
                   max_total_tasks_per_image: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, rejected = [], []
    per_format: Counter[tuple[str, str, str]] = Counter()
    total: Counter[tuple[str, str]] = Counter()
    for item in sorted(items, key=lambda value: (str(value.get("dataset_id")), str(value.get("item_id", value.get("id", ""))))):
        dataset = str(item.get("dataset_id", item.get("dataset", "")))
        keys = [(dataset, str(asset), str(item.get("output_spec"))) for asset in (item.get("input_assets") or [])]
        reasons = []
        if max_tasks_per_image is not None and any(per_format[key] >= max_tasks_per_image for key in keys):
            reasons.append("max_tasks_per_image")
        if max_total_tasks_per_image is not None and any(total[(key[0], key[1])] >= max_total_tasks_per_image for key in keys):
            reasons.append("max_total_tasks_per_image")
        if reasons:
            rejected.append({"item": item, "reasons": sorted(set(reasons))})
            continue
        accepted.append(item)
        for key in keys:
            per_format[key] += 1
            total[(key[0], key[1])] += 1
    return accepted, rejected


def _access_payload(access: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset_id", "revision", "annotation_digest", "image_manifest_digest", "profile_digest",
        "duckdb_digest", "mapping_digest", "dataset_card_digest", "image_manifest_digest",
        "split_manifest_digest", "sampling_manifest_digest",
    )
    return {key: access.get(key) for key in keys if access.get(key) is not None}


def _digest_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _verify_dataset_manifests(access: dict[str, Any]) -> dict[str, Any]:
    references = {}
    for name in ("split", "sampling"):
        path = access.get(f"{name}_manifest_path")
        digest = access.get(f"{name}_manifest_digest")
        if path or digest:
            if not path or not digest or not Path(path).is_file():
                raise ValueError(f"{name} manifest is missing")
            verify_manifest_digest(path, digest)
            model = {
                "split": SplitManifest,
                "sampling": SamplingManifest,
            }[name]
            load_manifest(path, model)
            references[name] = {"path": path, "digest": digest}
    return references


def _validate_item_manifest_refs(item: dict[str, Any], resolver_context) -> list[str]:
    errors = []
    sampling_id = item.get("sampling_unit_id")
    sampling = resolver_context.manifests.get("sampling") if resolver_context else None
    if sampling_id:
        unit = next((value for value in (sampling.units if sampling else []) if value.sampling_unit_id == sampling_id), None)
        if unit is None:
            errors.append("unknown_sampling_unit")
        else:
            if item.get("case_id") != unit.case_id:
                errors.append("sampling_case_mismatch")
            if item.get("split") != unit.split:
                errors.append("sampling_split_mismatch")
            if item.get("view_ids") != unit.view_ids:
                errors.append("sampling_view_order_mismatch")
            if item.get("panel_id") != unit.panel_id:
                errors.append("sampling_panel_mismatch")
            if item.get("sampling_order") != unit.order:
                errors.append("sampling_order_mismatch")
    elif sampling is not None and item.get("view_ids"):
        errors.append("missing_sampling_unit")
    return errors


def _manifest(context: dict[str, Any], plans: dict[str, TaskPlanState], allocation: dict[str, Any],
              items: list[dict[str, Any]], reports: dict[str, Any], option_distribution: dict[str, Any]) -> dict[str, Any]:
    tasks, datasets = {}, {}
    access_receipts = {}
    manifest_refs = {}
    for dataset_id in sorted(plans):
        plan = plans[dataset_id]
        access = context.get("dataset_assets", {}).get(dataset_id, {}).get("access", {})
        access_receipts[dataset_id] = _access_payload(access)
        manifest_refs[dataset_id] = _verify_dataset_manifests(access)
        counts = allocation["counts"].get(dataset_id, {})
        dataset_items = [item for item in items if item.get("dataset_id") == dataset_id]
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "plan_revision": plan.revision,
            "allocated_count": sum(counts.values()),
            "generated_count": reports[dataset_id]["generated"],
            "accepted_count": reports[dataset_id]["accepted"],
            "shortfall": reports[dataset_id]["shortfall"],
            "item_count": len(dataset_items),
            "dataset_access_digest": _digest_payload(access_receipts[dataset_id]),
            "prepared_annotation_digest": access.get("annotation_digest"),
            "image_manifest_digest": access.get("image_manifest_digest"),
            "split_manifest_digest": access.get("split_manifest_digest"),
            "sampling_manifest_digest": access.get("sampling_manifest_digest"),
        }
        for leaf_id, leaf in sorted(plan.leaf_tasks.items()):
            tasks[f"{dataset_id}::{leaf_id}"] = {
                "dataset_id": dataset_id,
                "leaf_task_id": leaf_id,
                "scenario_id": leaf.scenario_id,
                "capability_category": leaf.capability_category.value,
                "task_paradigm": leaf.task_paradigm.value,
                "output_spec": leaf.output_spec.value,
                "allocated_count": counts.get(leaf_id, 0),
                "eligible_count": leaf.eligible_count,
                "generated_count": sum(item.get("leaf_task_id") == leaf_id for item in dataset_items),
                "accepted_count": sum(item.get("leaf_task_id") == leaf_id for item in dataset_items),
            }
    digest_inputs = {
        "bir_intent": context.get("bir_intent", context.get("benchmark_info", {})),
        "registry_digest": context.get("registry_digest") or context.get("bir_manifest_digest"),
        "dataset_access": access_receipts,
        "task_plans": {
            dataset_id: plan.model_dump(mode="json")
            for dataset_id, plan in sorted(plans.items())
        },
        "task_plan_digests": context.get("task_plan_digests", {}),
        "presentation": {
            dataset_id: {
                name: value.get("digest")
                for name, value in refs.items()
            }
            for dataset_id, refs in manifest_refs.items()
        },
        "split": {dataset_id: access.get("split_manifest_digest") for dataset_id, access in (
            (key, context.get("dataset_assets", {}).get(key, {}).get("access", {})) for key in sorted(plans)
        )},
        "sampling": {dataset_id: access.get("sampling_manifest_digest") for dataset_id, access in (
            (key, context.get("dataset_assets", {}).get(key, {}).get("access", {})) for key in sorted(plans)
        )},
        "evaluation_protocol": context.get("evaluation_protocol", {}),
        "allocation": allocation,
        "seed": context.get("seed", 0),
    }
    return {
        "schema_version": "benchmark-manifest",
        "benchmark_id": _digest_payload(digest_inputs),
        "digest_inputs": digest_inputs,
        "benchmark_info": context.get("benchmark_info", {}),
        "dataset_access_digest": _digest_payload(access_receipts),
        "prepared_annotation_digest": {
            dataset_id: value.get("annotation_digest") for dataset_id, value in access_receipts.items()
        },
        "image_manifest_digest": {
            dataset_id: value.get("image_manifest_digest") for dataset_id, value in access_receipts.items()
        },
        "split_manifest_digest": {
            dataset_id: value.get("split_manifest_digest") for dataset_id, value in access_receipts.items()
        },
        "sampling_manifest_digest": {
            dataset_id: value.get("sampling_manifest_digest") for dataset_id, value in access_receipts.items()
        },
        "registry_digest": digest_inputs["registry_digest"],
        "task_plan_digest": context.get("task_plan_digests", {}),
        "evaluation_protocol": context.get("evaluation_protocol", {}),
        "view_panel_semantics": context.get("view_panel_semantics", {}),
        "sample_scope": context.get("sample_scope", []),
        "execution_limits": {
            "max_tasks_per_image": context.get("max_tasks_per_image"),
            "max_total_tasks_per_image": context.get("max_total_tasks_per_image"),
        },
        "datasets": datasets,
        "tasks": tasks,
        "allocation": allocation,
        "item_count": len(items),
        "generated_count": sum(report["generated"] for report in reports.values()),
        "option_distribution": option_distribution,
    }


def audit_benchmark(context: dict[str, Any], *, output_directory: str | Path | None = None) -> dict[str, Any]:
    store = BIRStore(context["bir_path"])
    bindings = store.load("dataset_bindings.json")["datasets"]
    selected = list(bindings)
    if len(selected) != 1:
        raise ValueError("Candidate audit requires exactly one dataset")
    allocation = store.load("allocation.json")
    assets = {key: {"access": value["access"].get("source_access", value["access"])} for key, value in bindings.items()}
    context = {**context, "dataset_assets": assets}
    plans = {
        dataset_id: TaskPlanState.model_validate(BIRStore(context["bir_path"]).load(f"datasets/{dataset_id}/planning/task_plan.json"))
        for dataset_id in selected
    }
    root = Path(output_directory or Path(context.get("run_directory", context["operation_directory"])) / "delivery")
    root.mkdir(parents=True, exist_ok=True)
    all_items, quarantine, reports = [], [], {}
    from medbenchagent.data.index import DatasetIndex

    for dataset_id in selected:
        counts = allocation.get("counts", {}).get(dataset_id, {})
        receipt = store.load(f"outputs/{dataset_id}.json")
        raw_path = receipt["benchmark_items_path"]
        from medbenchagent.data.sources import digest_file
        if digest_file(raw_path) != receipt["benchmark_items_digest"]:
            raise ValueError("Generation output digest changed")
        if not sum(counts.values()):
            reports[dataset_id] = {"generated": 0, "accepted": 0, "rejected": 0, "shortfall": 0, "status": "skipped"}
            continue
        if not raw_path or not Path(raw_path).is_file():
            raise ValueError(f"Generation output is missing for {dataset_id}")
        cases = list(DatasetIndex(context["dataset_assets"][dataset_id]["access"]).iter_cases())
        from medbenchagent.planning.resolver import ResolverContext
        resolver_context = ResolverContext.from_access(context["dataset_assets"][dataset_id]["access"])
        resolver_context.scenarios = plans[dataset_id].scenarios
        _verify_dataset_manifests(context["dataset_assets"][dataset_id]["access"])
        raw_items = _read_jsonl(raw_path)
        raw_generated_count = len(raw_items)
        for row in raw_items:
            if row.get("__invalid__"):
                quarantine.append({"dataset_id": dataset_id, "stage": "generation", "item": row, "reasons": ["invalid_json"]})
        generated = []
        for row in raw_items:
            if row.get("__invalid__"):
                continue
            if row.get("dataset_id", row.get("dataset")) != dataset_id or row.get("dataset") not in {None, dataset_id}:
                quarantine.append({"dataset_id": dataset_id, "stage": "audit", "item": row, "reasons": ["dataset_mismatch"]})
                continue
            manifest_reasons = _validate_item_manifest_refs(row, resolver_context)
            if manifest_reasons:
                quarantine.append({"dataset_id": dataset_id, "stage": "manifest", "item": row, "reasons": manifest_reasons})
                continue
            generated.append(row)
        audit = audit_task_items(
            cases, generated, plans[dataset_id].leaf_tasks.values(),
            resolver_context=resolver_context,
        )
        accepted = list(audit["accepted_items"])
        quarantine.extend({"dataset_id": dataset_id, "stage": "audit", **row} for row in audit["rejected_items"])
        bounded, over_allocated = [], []
        allocated_seen: Counter[str] = Counter()
        for item in sorted(accepted, key=lambda value: str(value.get("item_id", value.get("id", "")))):
            leaf_id = str(item.get("leaf_task_id", item.get("task_id", "")))
            if allocated_seen[leaf_id] >= counts.get(leaf_id, 0):
                over_allocated.append({"item": item, "reasons": ["allocation_exceeded"]})
            else:
                allocated_seen[leaf_id] += 1
                bounded.append(item)
        accepted = bounded
        quarantine.extend({"dataset_id": dataset_id, "stage": "assembly", **row} for row in over_allocated)
        seen, unique = set(), []
        for item in accepted:
            identity = (dataset_id, item.get("item_id", item.get("id")))
            if identity in seen:
                quarantine.append({"dataset_id": dataset_id, "stage": "assembly", "item": item, "reasons": ["duplicate_item_id"]})
            else:
                seen.add(identity)
                unique.append(item)
        accepted, limited = _limit_reasons(
            unique,
            max_tasks_per_image=context.get("max_tasks_per_image"),
            max_total_tasks_per_image=context.get("max_total_tasks_per_image"),
        )
        quarantine.extend({"dataset_id": dataset_id, "stage": "assembly", **row} for row in limited)
        all_items.extend(accepted)
        allocated = sum(counts.values())
        reports[dataset_id] = {
            "generated": raw_generated_count, "accepted": len(accepted), "rejected": raw_generated_count - len(accepted),
            "shortfall": max(0, allocated - len(accepted)), "allocated": allocated, "status": "complete",
            "audit": {key: value for key, value in audit.items() if key not in {"accepted_items", "rejected_items"}},
        }

    all_items, option_summary = balance_option_positions(all_items, context.get("seed", 0))
    if not all_items:
        raise ValueError("No deliverable benchmark items remain after Finalization")
    manifest = _manifest(context, plans, allocation, all_items, reports, option_summary)
    report = {
        "schema_version": "audit-report",
        "generated": sum(value["generated"] for value in reports.values()),
        "accepted": len(all_items), "rejected": len(quarantine),
        "requested": allocation.get("requested_count", 0),
        "shortfall": max(0, allocation.get("requested_count", 0) - len(all_items)),
        "datasets": reports, "option_distribution": option_summary, "quarantine_count": len(quarantine),
    }
    paths = {
        "delivery_path": str(root),
        "benchmark_path": _write_jsonl(root / "benchmark.jsonl", all_items),
        "manifest_path": str(root / "benchmark_manifest.json"),
        "audit_report_path": str(root / "audit_report.json"),
        "quarantine_path": _write_jsonl(root / "quarantine.jsonl", quarantine),
    }
    atomic_write_json(paths["manifest_path"], manifest)
    atomic_write_json(paths["audit_report_path"], report)
    return result("success", f"Audited {len(all_items)} candidate benchmark items",
                  **paths, delivery_counts=report, item_count=len(all_items))
