"""Deterministic audit and reporting for benchmark items."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.planning.bir import BIRStore
from typing import Any, Iterable

from medbenchagent.generation.items import _input_assets, _input_bindings
from medbenchagent.planning.resolver import NotApplicable, ResolverContext, encode_closed_label, resolve_task_input
from pydantic import ValidationError
from medbenchagent.contracts import BenchmarkItem
from medbenchagent.planning.task_space import LeafTask, OutputSpec


def _answer_matches(spec: OutputSpec, expected: Any, actual: Any) -> bool:
    if spec == OutputSpec.NUMBER:
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == expected
    if spec == OutputSpec.LABELS:
        return isinstance(actual, list) and isinstance(expected, list) and sorted(actual) == sorted(expected)
    if spec == OutputSpec.BBOX:
        return isinstance(actual, list) and len(actual) == 4 and actual == expected
    if spec == OutputSpec.INDEX:
        return type(actual) is int and actual >= 0 and actual == expected
    if spec == OutputSpec.LABEL:
        return type(actual) in (str, int, bool) and actual == expected
    if spec == OutputSpec.TEXT:
        return isinstance(actual, str) and actual == expected
    return False


def audit_task_items(
    cases: Iterable[dict[str, Any]],
    items: Iterable[dict[str, Any]],
    leaves: Iterable[LeafTask | dict[str, Any]],
    resolver_context: ResolverContext | None = None,
) -> dict[str, Any]:
    items = list(items)
    leaf_map = {leaf.leaf_task_id: leaf for leaf in (LeafTask.model_validate(value) for value in leaves)}
    case_map = {case.get("case_id"): case for case in cases}
    accepted, rejected = [], []
    reasons = Counter()
    for item in items:
        try:
            BenchmarkItem.model_validate(item)
        except ValidationError as exc:
            reasons.update(["invalid_item_contract"])
            rejected.append({"item": item, "reasons": ["invalid_item_contract"],
                             "diagnostics": str(exc)})
            continue
        leaf = leaf_map.get(item.get("leaf_task_id"))
        case = case_map.get(item.get("case_id"))
        errors = []
        if leaf is None:
            errors.append("unknown_leaf_task")
        if case is None:
            errors.append("unknown_case")
        if not errors:
            sampling_unit = None
            if resolver_context and resolver_context.manifests.get("sampling") and not item.get("sampling_unit_id"):
                errors.append("missing_sampling_unit")
            if item.get("sampling_unit_id"):
                manifest = (resolver_context.manifests.get("sampling") if resolver_context else None)
                sampling_unit = next(
                    (unit for unit in (manifest.units if manifest is not None else [])
                     if unit.sampling_unit_id == item.get("sampling_unit_id")),
                    None,
                )
                if sampling_unit is None:
                    errors.append("unknown_sampling_unit")
                elif sampling_unit.case_id != case.get("case_id") or not sampling_unit.eligible:
                    errors.append("invalid_sampling_unit")
            resolved = resolve_task_input(case, leaf, sampling_unit, resolver_context) if not errors else None
            if isinstance(resolved, NotApplicable):
                errors.append(resolved.reason)
            elif resolved is not None:
                expected = resolved.reference.get("value", resolved.reference) if isinstance(resolved.reference, dict) else resolved.reference
                if leaf.output_spec == OutputSpec.LABEL and leaf.options is not None:
                    try:
                        expected = encode_closed_label(expected)
                    except ValueError:
                        errors.append("invalid_label_encoding")
                if item.get("input_images") != resolved.presentations:
                    errors.append("input_presentation_mismatch")
                expected_bindings = _input_bindings(case, resolved.input_bindings)
                template_index = item.get("template_index", -1)
                expected_question = (
                    leaf.question_templates[template_index]
                    if isinstance(template_index, int) and 0 <= template_index < len(leaf.question_templates)
                    else None
                )
                if item.get("question") != expected_question:
                    errors.append("question_mismatch")
                if item.get("template_count") != len(leaf.question_templates):
                    errors.append("template_count_mismatch")
                if not isinstance(template_index, int) or template_index < 0 or template_index >= len(leaf.question_templates):
                    errors.append("template_index_invalid")
                if item.get("options") != leaf.options:
                    errors.append("options_mismatch")
                if item.get("input_bindings") != expected_bindings:
                    errors.append("input_binding_mismatch")
                if item.get("input_evidence_ids") != resolved.input_evidence_ids:
                    errors.append("input_order_mismatch")
                if item.get("input_assets") != _input_assets(case, resolved.input_evidence_ids):
                    errors.append("input_asset_order_mismatch")
                if item.get("reference_evidence_ids") != resolved.reference_evidence_ids:
                    errors.append("reference_evidence_mismatch")
                if sampling_unit is not None:
                    if item.get("split") != sampling_unit.split:
                        errors.append("sampling_split_mismatch")
                    if item.get("view_ids") != sampling_unit.view_ids:
                        errors.append("sampling_view_order_mismatch")
                    if item.get("panel_id") != sampling_unit.panel_id:
                        errors.append("sampling_panel_mismatch")
                    if item.get("sampling_order") != sampling_unit.order:
                        errors.append("sampling_order_mismatch")
                if not _answer_matches(leaf.output_spec, expected, item.get("answer")):
                    errors.append("answer_mismatch")
                if set(item.get("input_evidence_ids", [])) & set(item.get("reference_evidence_ids", [])):
                    errors.append("input_reference_leakage")
        if errors:
            reasons.update(errors)
            rejected.append({"item": item, "reasons": errors})
        else:
            accepted.append(item)
    by_capability = defaultdict(lambda: {"generated": 0, "accepted": 0, "rejected": 0})
    for leaf_id, leaf in leaf_map.items():
        generated_count = sum(isinstance(item, dict) and item.get("leaf_task_id") == leaf_id for item in items)
        accepted_count = sum(item.get("leaf_task_id") == leaf_id for item in accepted)
        group = by_capability[leaf.capability_category.value]
        group["generated"] += generated_count
        group["accepted"] += accepted_count
        group["rejected"] += generated_count - accepted_count
    return {
        "schema_version": "audit-report",
        "generated": len(items),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reason_counts": dict(reasons),
        "capability_categories": dict(by_capability),
        "accepted_items": accepted,
        "rejected_items": rejected,
    }



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
            distribution[str(item.get("leaf_task_id"))][str(options.index(item["answer"]))] += 1
    return {task: dict(sorted(counts.items())) for task, counts in sorted(distribution.items())}


def balance_option_positions(items: list[dict[str, Any]], seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance answer positions while preserving semantic answer values."""
    before = _option_distribution(items)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        if isinstance(item.get("options"), list) and item.get("options") and item.get("answer") in item["options"]:
            groups[str(item.get("leaf_task_id"))].append(item)
    for task_id, group in groups.items():
        group.sort(key=lambda item: hashlib.sha256(
            json.dumps([seed, task_id, item["item_id"]], separators=(",", ":")).encode()
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
    for item in sorted(items, key=lambda value: (str(value.get("dataset_id")), str(value["item_id"]))):
        dataset = str(item["dataset_id"])
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


def audit_benchmark(context, *, output_directory):
    from medbenchagent.data.index import DatasetIndex
    from medbenchagent.data.manifests import load_manifest, SamplingManifest, SplitManifest
    from medbenchagent.data.sources import digest_file
    store = BIRStore(context["bir_path"])
    access = store.load("dataset_binding.json")["access"]
    dataset = access["dataset_id"]
    index = DatasetIndex(access)
    for name, model in (("sampling", SamplingManifest), ("split", SplitManifest)):
        if path := access.get(f"{name}_manifest_path"):
            load_manifest(path, model)
    plan = TaskPlanState.model_validate(store.load("task_plan.json"))
    allocation = store.load("allocation.json")
    output = store.load("generation.json")
    raw_path = output["benchmark_items_path"]
    if digest_file(raw_path) != output["benchmark_items_digest"]:
        raise ValueError("Generation output digest changed")
    resolver = ResolverContext.from_access(access)
    resolver.scenarios = plan.scenarios
    raw = _read_jsonl(raw_path)
    valid, quarantine = [], []
    for item in raw:
        try:
            parsed = BenchmarkItem.model_validate(item)
            if parsed.dataset_id != dataset:
                raise ValueError("dataset_mismatch")
        except (ValueError, ValidationError) as exc:
            quarantine.append({"item": item, "reasons": ["invalid_item_contract"], "diagnostics": str(exc)})
        else:
            valid.append(item)
    checked = audit_task_items(index.iter_cases(), valid, plan.leaf_tasks.values(), resolver_context=resolver)
    quarantine.extend(checked.pop("rejected_items"))
    accepted, seen, counts = [], set(), Counter()
    for item in sorted(checked.pop("accepted_items"), key=lambda row: row["item_id"]):
        reason = ("duplicate_item_id" if item["item_id"] in seen else
                  "allocation_exceeded" if counts[item["leaf_task_id"]] >= allocation["counts"].get(item["leaf_task_id"], 0) else None)
        if reason:
            quarantine.append({"item": item, "reasons": [reason]})
            continue
        seen.add(item["item_id"])
        counts[item["leaf_task_id"]] += 1
        accepted.append(item)
    accepted, limited = _limit_reasons(accepted, max_tasks_per_image=context.get("max_tasks_per_image"),
                                      max_total_tasks_per_image=context.get("max_total_tasks_per_image"))
    quarantine.extend(limited)
    accepted, options = balance_option_positions(accepted, context.get("seed", 0))
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    quarantine_path = _write_jsonl(root / "quarantine.jsonl", quarantine)
    reasons = dict(Counter(reason for row in quarantine for reason in row["reasons"]))
    if not accepted:
        error = ValueError(f"No deliverable benchmark items remain after candidate audit; inspect {quarantine_path}")
        error.diagnostics = {"quarantine_path": quarantine_path, "reason_counts": reasons}
        raise error
    identity = {"binding": store.digest("dataset_binding.json"), "plan": store.digest("task_plan.json"),
                "allocation": store.digest("allocation.json"), "generation": store.digest("generation.json"),
                "seed": context.get("seed", 0), "max_tasks_per_image": context.get("max_tasks_per_image"),
                "max_total_tasks_per_image": context.get("max_total_tasks_per_image")}
    tasks = {f"{dataset}::{key}": {"dataset_id": dataset, "leaf_task_id": key, "scenario_id": leaf.scenario_id,
             "capability_category": leaf.capability_category.value, "task_paradigm": leaf.task_paradigm.value,
             "output_spec": leaf.output_spec.value, "eligible_count": leaf.eligible_count,
             "allocated_count": allocation["counts"].get(key, 0),
             "accepted_count": sum(item["leaf_task_id"] == key for item in accepted)} for key, leaf in plan.leaf_tasks.items()}
    manifest = {"schema_version": "benchmark-manifest", "dataset_id": dataset,
                "benchmark_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "dependencies": identity, "benchmark_info": store.load("benchmark_info.json"),
                "tasks": tasks, "allocation": allocation, "item_count": len(accepted), "generated_count": len(raw)}
    report = {"schema_version": "audit-report", "generated": len(raw), "accepted": len(accepted),
              "rejected": len(quarantine), "requested": allocation["requested_count"],
              "shortfall": max(0, allocation["requested_count"] - len(accepted)),
              "reason_counts": reasons, "item_audit": checked, "option_distribution": options}
    from medbenchagent.generation.items import write_task_items
    write_task_items(root / "benchmark.jsonl", accepted)
    atomic_write_json(root / "benchmark_manifest.json", manifest)
    atomic_write_json(root / "audit_report.json", report)
    return result("success", f"Audited {len(accepted)} accepted items", data={"counts": report, "directory": str(root)})
