"""Deterministic global Finalization stage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.audit.task_items import audit_task_items
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json


PACKAGE_FILES = ("benchmark.jsonl", "benchmark_manifest.json", "audit_report.json", "quarantine.jsonl")


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


def _manifest(context: dict[str, Any], plans: dict[str, TaskPlanState], allocation: dict[str, Any],
              items: list[dict[str, Any]], reports: dict[str, Any], option_distribution: dict[str, Any]) -> dict[str, Any]:
    tasks, datasets = {}, {}
    for dataset_id in sorted(plans):
        plan = plans[dataset_id]
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
    return {
        "schema_version": "benchmark-manifest",
        "benchmark_id": hashlib.sha256(json.dumps(
            [sorted(plans), allocation, context.get("seed", 0)], sort_keys=True, ensure_ascii=False
        ).encode()).hexdigest(),
        "benchmark_info": context.get("benchmark_info", {}),
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


def finalize_benchmark(context: dict[str, Any], *, output_directory: str | Path | None = None) -> dict[str, Any]:
    selected = list(context.get("selected_dataset_ids", []))
    allocation_path = context.get("allocation_path")
    if not selected or not allocation_path or not Path(allocation_path).is_file():
        raise ValueError("Dataset selection and global allocation are required before finalization")
    allocation = json.loads(Path(allocation_path).read_text(encoding="utf-8"))
    plans = {
        dataset_id: TaskPlanState.model_validate(json.loads(Path(context["dataset_task_plans"][dataset_id]).read_text(encoding="utf-8")))
        for dataset_id in selected
    }
    root = Path(output_directory or Path(context.get("run_directory", context["operation_directory"])) / "delivery")
    root.mkdir(parents=True, exist_ok=True)
    all_items, quarantine, reports = [], [], {}
    from medbenchagent.data.index import DatasetIndex

    for dataset_id in selected:
        counts = allocation.get("counts", {}).get(dataset_id, {})
        generation = context.get("dataset_generations", {}).get(dataset_id, {})
        raw_path = generation.get("raw_output_path") if isinstance(generation, dict) else generation
        if not sum(counts.values()):
            reports[dataset_id] = {"generated": 0, "accepted": 0, "rejected": 0, "shortfall": 0, "status": "skipped"}
            continue
        if not raw_path or not Path(raw_path).is_file():
            raise ValueError(f"Generation output is missing for {dataset_id}")
        cases = list(DatasetIndex(context["dataset_assets"][dataset_id]["access"]).iter_cases())
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
            generated.append(row)
        audit = audit_task_items(cases, generated, plans[dataset_id].leaf_tasks.values())
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
    atomic_write_json(root / "complete.json", {"schema_version": "benchmark-delivery", "files": list(PACKAGE_FILES)})
    return result("success", f"Finalized {len(all_items)} benchmark items from {len(selected)} datasets",
                  **paths, delivery_counts=report, item_count=len(all_items))


def restore_delivery(context):
    return {"delivery_path": context.get("delivery_path"), "delivery_counts": context.get("delivery_counts", {})}


def compile_finalization_agent(*, checkpointer=None):
    return finalize_benchmark
