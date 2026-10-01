"""Deterministic benchmark item generation for compiled v2 Task Leaves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
from medbenchagent.planning.task_space import OutputSpec, TaskLeaf


def _identity(dataset_id: str, leaf_id: str, case_id: str) -> str:
    payload = json.dumps([dataset_id, leaf_id, case_id], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _input_assets(case: dict[str, Any], evidence_ids: list[str]) -> list[str]:
    wanted = set(evidence_ids)
    return [str(item["value"]) for item in case.get("evidence", [])
            if item.get("id") in wanted and item.get("type") == "asset"]


def generate_task_items(cases: Iterable[dict[str, Any]], leaves: Iterable[TaskLeaf | dict[str, Any]], dataset_id: str) -> list[dict[str, Any]]:
    """Instantiate eligible Cases exactly once per Leaf using its compiled Resolver."""
    parsed = [TaskLeaf.model_validate(leaf) for leaf in leaves]
    items = []
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        for leaf in parsed:
            resolved = resolve(case, ResolverSpec.parse(leaf.resolver))
            if isinstance(resolved, NotApplicable):
                continue
            answer = resolved.reference
            if leaf.output_spec == OutputSpec.TEXT and isinstance(answer, dict):
                answer = answer.get("value")
            item = {
                "id": _identity(dataset_id, leaf.leaf_id, case_id),
                "dataset": dataset_id,
                "task_id": leaf.leaf_id,
                "case_id": case_id,
                "input_assets": _input_assets(case, resolved.input_evidence_ids),
                "input_evidence_ids": resolved.input_evidence_ids,
                "question": leaf.question,
                "output_spec": leaf.output_spec.value,
                "answer": answer,
                "reference_evidence_ids": resolved.reference_evidence_ids,
            }
            items.append(item)
    return items


def write_task_items(path: str | Path, items: Iterable[dict[str, Any]]) -> str:
    """Write v2 items atomically as JSONL and return the resulting path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return str(destination)

