"""Deterministic benchmark item generation for Leaf Tasks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.planning.resolver import NotApplicable, ResolverContext, ResolverSpec, encode_closed_label, resolve
from medbenchagent.planning.task_space import LeafTask, OutputSpec
from medbenchagent.data.manifests import SamplingManifest, SamplingUnit


def _identity(
    dataset_id: str,
    leaf_id: str,
    case_id: str,
    sampling_unit_id: str | None = None,
    view_ids: list[str] | None = None,
    panel_id: str | None = None,
    order: int | None = None,
) -> str:
    payload = json.dumps(
        [dataset_id, leaf_id, case_id, sampling_unit_id, view_ids or [], panel_id, order],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _input_assets(case: dict[str, Any], evidence_ids: list[str]) -> list[str]:
    by_id = {item.get("id"): item for item in case.get("evidence", [])}
    return [
        str(by_id[evidence_id]["value"])
        for evidence_id in evidence_ids
        if evidence_id in by_id and by_id[evidence_id].get("type") == "asset"
    ]


def _input_bindings(case: dict[str, Any], bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item.get("id"): item for item in case.get("evidence", [])}
    return [
        {
            "role": binding["role"],
            "evidence_ids": list(binding["evidence_ids"]),
            "assets": [
                str(by_id[evidence_id]["value"])
                for evidence_id in binding["evidence_ids"]
                if evidence_id in by_id and by_id[evidence_id].get("type") == "asset"
            ],
        }
        for binding in bindings
    ]


def _case_for_unit(case: dict[str, Any], unit: SamplingUnit) -> dict[str, Any]:
    """Return a shallow Case view with manifest order applied to Evidence."""
    if not unit.ordered_evidence_ids and not unit.view_ids:
        return case
    evidence = list(case.get("evidence", []))
    evidence_order = {identifier: index for index, identifier in enumerate(unit.ordered_evidence_ids)}
    view_order = {identifier: index for index, identifier in enumerate(unit.view_ids)}
    original = {id(item): index for index, item in enumerate(evidence)}

    def key(item):
        if item.get("id") in evidence_order:
            return (0, evidence_order[item["id"]])
        if item.get("view_id") in view_order:
            return (1, view_order[item["view_id"]])
        return (2, original[id(item)])

    return {**case, "evidence": sorted(evidence, key=key)}


def _sampling_units(
    cases: list[dict[str, Any]],
    sampling_manifest: SamplingManifest | dict[str, Any] | Iterable[SamplingUnit | dict[str, Any]] | None,
) -> list[tuple[dict[str, Any], SamplingUnit]]:
    by_case = {case.get("case_id"): case for case in cases if isinstance(case.get("case_id"), str)}
    if sampling_manifest is None:
        return [
            (case, SamplingUnit(
                sampling_unit_id=str(case["case_id"]), case_id=str(case["case_id"]), split="unspecified",
                view_ids=[], ordered_evidence_ids=[], panel_id=None, eligible=True, reason=None, order=index,
            ))
            for index, case in enumerate(cases)
            if isinstance(case.get("case_id"), str) and case.get("case_id")
        ]
    if isinstance(sampling_manifest, SamplingManifest):
        units = sampling_manifest.units
    elif isinstance(sampling_manifest, dict):
        units = SamplingManifest.model_validate(sampling_manifest).units
    else:
        units = [SamplingUnit.model_validate(unit) for unit in sampling_manifest]
    result = []
    for unit in sorted(units, key=lambda item: item.order):
        case = by_case.get(unit.case_id)
        if case is not None and unit.eligible:
            result.append((_case_for_unit(case, unit), unit))
    return result


def _template_cycle(seed: int, dataset_id: str, leaf_id: str, count: int) -> list[int]:
    payload = json.dumps([seed, dataset_id, leaf_id], separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    cycle = list(range(count))
    rng.shuffle(cycle)
    return cycle


def generate_task_items(
    cases: Iterable[dict[str, Any]],
    leaves: Iterable[LeafTask | dict[str, Any]],
    dataset_id: str,
    allocation: dict[str, int] | None = None,
    seed: int = 0,
    resolver_context: ResolverContext | None = None,
    sampling_manifest: SamplingManifest | dict[str, Any] | Iterable[SamplingUnit | dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    parsed = [LeafTask.model_validate(leaf) for leaf in leaves]
    items = []
    limits = allocation or {leaf.leaf_task_id: None for leaf in parsed}
    counts = {leaf.leaf_task_id: 0 for leaf in parsed}
    cycles = {
        leaf.leaf_task_id: _template_cycle(seed, dataset_id, leaf.leaf_task_id, len(leaf.question_templates))
        for leaf in parsed
    }
    case_list = list(cases)
    volume_input = False
    if resolver_context is not None:
        volume_input = any(
            resolver_context.asset(str(item.get("value")))
            and resolver_context.asset(str(item.get("value"))).get("kind") == "volume"
            for case in case_list
            for item in case.get("evidence", [])
            if item.get("type") == "asset"
        )
    # A prepared Case may carry view identities even when a Leaf consumes one
    # ordinary image.  Those identities alone do not make generation
    # view-dependent: only multi-input Leaves and volume inputs need a frozen
    # sampling order to bind their model inputs deterministically.
    requires_manifest = volume_input or any(len(leaf.input) > 1 for leaf in parsed)
    if requires_manifest and sampling_manifest is None:
        raise ValueError("View-dependent Generation requires a SamplingManifest")
    for case, sampling_unit in _sampling_units(case_list, sampling_manifest):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        for leaf in parsed:
            limit = limits.get(leaf.leaf_task_id)
            if limit is not None and counts[leaf.leaf_task_id] >= limit:
                continue
            resolved = resolve(case, ResolverSpec.parse(leaf.resolver), resolver_context=resolver_context)
            if isinstance(resolved, NotApplicable):
                continue
            answer = resolved.reference
            if leaf.output_spec == OutputSpec.TEXT and isinstance(answer, dict):
                answer = answer.get("value")
            if leaf.output_spec == OutputSpec.LABEL and leaf.options is not None:
                answer = encode_closed_label(answer.get("value", answer) if isinstance(answer, dict) else answer)
                if answer not in leaf.options:
                    raise ValueError(
                        f"Resolved label {answer!r} is missing from Leaf Task options for {leaf.leaf_task_id}"
                    )
            template_index = cycles[leaf.leaf_task_id][counts[leaf.leaf_task_id] % len(cycles[leaf.leaf_task_id])]
            item = {
                "item_id": _identity(dataset_id, leaf.leaf_task_id, case_id, sampling_unit.sampling_unit_id, sampling_unit.view_ids, sampling_unit.panel_id, sampling_unit.order),
                "id": _identity(dataset_id, leaf.leaf_task_id, case_id, sampling_unit.sampling_unit_id, sampling_unit.view_ids, sampling_unit.panel_id, sampling_unit.order),
                "dataset_id": dataset_id,
                "dataset": dataset_id,
                "scenario_id": leaf.scenario_id,
                "leaf_task_id": leaf.leaf_task_id,
                "task_id": leaf.leaf_task_id,
                "case_id": case_id,
                "capability_category": leaf.capability_category.value,
                "task_paradigm": leaf.task_paradigm.value,
                "options": list(leaf.options) if leaf.options is not None else None,
                "input_bindings": _input_bindings(case, resolved.input_bindings),
                "input_assets": _input_assets(case, resolved.input_evidence_ids),
                "input_evidence_ids": resolved.input_evidence_ids,
                "question": leaf.question_templates[template_index],
                "template_index": template_index,
                "template_count": len(leaf.question_templates),
                "output_spec": leaf.output_spec.value,
                "answer": answer,
                "reference_evidence_ids": resolved.reference_evidence_ids,
            }
            if sampling_manifest is not None:
                item.update({
                    "sampling_unit_id": sampling_unit.sampling_unit_id,
                    "split": sampling_unit.split,
                    "view_ids": list(sampling_unit.view_ids),
                    "panel_id": sampling_unit.panel_id,
                    "sampling_order": sampling_unit.order,
                })
            items.append(item)
            counts[leaf.leaf_task_id] += 1
    return items


def write_task_items(path: str | Path, items: Iterable[dict[str, Any]]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return str(destination)

