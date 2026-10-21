"""Deterministic audit and reporting for benchmark items."""

from __future__ import annotations

from collections import Counter, defaultdict
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

