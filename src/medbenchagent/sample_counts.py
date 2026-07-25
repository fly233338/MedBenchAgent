"""Deterministic per-task sample-count planning for working Blueprints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medbenchagent.settings import get_settings


def _category_weights() -> dict[str, float]:
    return dict(get_settings().category_weights)


def distribute_sample_counts(
    task_ids: list[str], total: int, fixed_counts: dict[str, int] | None = None
) -> dict[str, int]:
    """Keep explicit counts and distribute the remainder in task order."""
    if not isinstance(total, int) or total <= 0:
        raise ValueError("Benchmark sample_count must be a positive integer.")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Task IDs must be non-empty and unique.")
    fixed = dict(fixed_counts or {})
    if unknown := set(fixed) - set(task_ids):
        raise ValueError(f"Unknown task IDs in sample counts: {', '.join(sorted(unknown))}.")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in fixed.values()):
        raise ValueError("Task sample counts must be non-negative integers.")
    fixed_total = sum(fixed.values())
    if fixed_total > total:
        raise ValueError("Fixed task sample counts exceed the benchmark sample_count.")
    remaining_ids = [task_id for task_id in task_ids if task_id not in fixed]
    remaining_total = total - fixed_total
    if not remaining_ids:
        if remaining_total:
            raise ValueError("Complete task sample counts must sum to the benchmark sample_count.")
        return {task_id: fixed[task_id] for task_id in task_ids}

    quotient, remainder = divmod(remaining_total, len(remaining_ids))
    distributed = dict(fixed)
    for index, task_id in enumerate(remaining_ids):
        distributed[task_id] = quotient + (index < remainder)
    return {task_id: distributed[task_id] for task_id in task_ids}


def _validate_weighted_tasks(tasks: dict[str, dict[str, Any]]) -> None:
    weights = _category_weights()
    if not tasks:
        raise ValueError("Blueprint tasks must be non-empty.")
    for task_id, task in tasks.items():
        category = task.get("category") if isinstance(task, dict) else None
        if category not in weights:
            raise ValueError(f"{task_id}: unsupported task category: {category!r}.")


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("Category weights must sum to a positive value.")
    quotas = {category: total * weight / weight_total for category, weight in weights.items()}
    allocated = {category: int(quota) for category, quota in quotas.items()}
    for category in sorted(
        weights,
        key=lambda category: (-(quotas[category] - allocated[category]), list(weights).index(category)),
    )[: total - sum(allocated.values())]:
        allocated[category] += 1
    return allocated


def distribute_weighted_sample_counts(
    tasks: dict[str, dict[str, Any]], total: int, fixed_counts: dict[str, int] | None = None
) -> dict[str, int]:
    """Allocate remaining samples by active category weight, then evenly within each category."""
    if not isinstance(total, int) or total <= 0:
        raise ValueError("Benchmark sample_count must be a positive integer.")
    _validate_weighted_tasks(tasks)
    fixed = dict(fixed_counts or {})
    if unknown := set(fixed) - set(tasks):
        raise ValueError(f"Unknown task IDs in sample counts: {', '.join(sorted(unknown))}.")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in fixed.values()):
        raise ValueError("Task sample counts must be non-negative integers.")
    remaining_total = total - sum(fixed.values())
    if remaining_total < 0:
        raise ValueError("Fixed task sample counts exceed the benchmark sample_count.")
    remaining_by_category: dict[str, list[str]] = {}
    for task_id, task in tasks.items():
        if task_id not in fixed:
            remaining_by_category.setdefault(task["category"], []).append(task_id)
    if not remaining_by_category:
        if remaining_total:
            raise ValueError("Complete task sample counts must sum to the benchmark sample_count.")
        return {task_id: fixed[task_id] for task_id in tasks}

    weights = _category_weights()
    active_weights = {
        category: weights[category]
        for category in weights
        if category in remaining_by_category
    }
    category_counts = _largest_remainder(remaining_total, active_weights)
    distributed = dict(fixed)
    for category, task_ids in remaining_by_category.items():
        quotient, remainder = divmod(category_counts[category], len(task_ids))
        for index, task_id in enumerate(task_ids):
            distributed[task_id] = quotient + (index < remainder)
    return {task_id: distributed[task_id] for task_id in tasks}


def write_sample_counts(blueprint_path: str | Path, counts: dict[str, int]) -> None:
    """Atomically persist confirmed task counts in the working Blueprint."""
    path = Path(blueprint_path)
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(counts):
        raise ValueError("Sample counts must contain exactly every retained Blueprint task.")
    total = blueprint.get("benchmark_info", {}).get("sample_count")
    normalized = distribute_weighted_sample_counts(tasks, total, counts)
    for task_id, count in normalized.items():
        task = tasks[task_id]
        if not isinstance(task, dict):
            raise ValueError(f"Blueprint task must be an object: {task_id}.")
        task["sample_count"] = count

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
