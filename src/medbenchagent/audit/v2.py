"""Deterministic audit and reporting for v2 benchmark items."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
from medbenchagent.planning.task_space import OutputSpec, TaskLeaf


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


def audit_task_items(cases: Iterable[dict[str, Any]], items: Iterable[dict[str, Any]], leaves: Iterable[TaskLeaf | dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct every reference and return accepted/rejected counts and reasons."""
    items = list(items)
    leaf_map = {leaf.leaf_id: leaf for leaf in (TaskLeaf.model_validate(value) for value in leaves)}
    case_map = {case.get("case_id"): case for case in cases}
    accepted, rejected = [], []
    reasons = Counter()
    for item in items:
        leaf = leaf_map.get(item.get("task_id"))
        case = case_map.get(item.get("case_id"))
        errors = []
        if leaf is None:
            errors.append("unknown_task_leaf")
        if case is None:
            errors.append("unknown_case")
        if not errors:
            resolved = resolve(case, ResolverSpec.parse(leaf.resolver))
            if isinstance(resolved, NotApplicable):
                errors.append(resolved.reason)
            else:
                if not _answer_matches(leaf.output_spec, resolved.reference.get("value", resolved.reference) if isinstance(resolved.reference, dict) else resolved.reference, item.get("answer")):
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
        group = by_capability[leaf.capability_category]
        group["generated"] += sum(item.get("task_id") == leaf_id for item in items)
        group["accepted"] += sum(item.get("task_id") == leaf_id for item in accepted)
        group["rejected"] += sum(item.get("task_id") == leaf_id for item in items) - group["accepted"]
    return {"schema_version": "audit-report.v2", "generated": len(items),
            "accepted": len(accepted), "rejected": len(rejected), "reason_counts": dict(reasons),
            "capability_categories": dict(by_capability), "accepted_items": accepted, "rejected_items": rejected}

