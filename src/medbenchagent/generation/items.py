"""Deterministic benchmark item generation for Leaf Tasks."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.planning.resolver import NotApplicable, ResolverContext, ResolverSpec, encode_closed_label, resolve
from medbenchagent.planning.task_space import LeafTask, OutputSpec


def _identity(dataset_id: str, leaf_id: str, case_id: str) -> str:
    payload = json.dumps([dataset_id, leaf_id, case_id], separators=(",", ":"))
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
) -> list[dict[str, Any]]:
    parsed = [LeafTask.model_validate(leaf) for leaf in leaves]
    items = []
    limits = allocation or {leaf.leaf_id: None for leaf in parsed}
    counts = {leaf.leaf_id: 0 for leaf in parsed}
    cycles = {
        leaf.leaf_id: _template_cycle(seed, dataset_id, leaf.leaf_id, len(leaf.question_templates))
        for leaf in parsed
    }
    ordered_cases = sorted(cases, key=lambda value: str(value.get("case_id", "")))
    for case in ordered_cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        for leaf in parsed:
            limit = limits.get(leaf.leaf_id)
            if limit is not None and counts[leaf.leaf_id] >= limit:
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
                        f"Resolved label {answer!r} is missing from Leaf Task options for {leaf.leaf_id}"
                    )
            template_index = cycles[leaf.leaf_id][counts[leaf.leaf_id] % len(cycles[leaf.leaf_id])]
            items.append({
                "item_id": _identity(dataset_id, leaf.leaf_id, case_id),
                "id": _identity(dataset_id, leaf.leaf_id, case_id),
                "dataset_id": dataset_id,
                "dataset": dataset_id,
                "scenario_id": leaf.scenario_id,
                "leaf_task_id": leaf.leaf_id,
                "task_id": leaf.leaf_id,
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
            })
            counts[leaf.leaf_id] += 1
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

