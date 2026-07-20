"""Deterministic per-task sample-count planning for working Blueprints."""

from __future__ import annotations

import json
from pathlib import Path


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


def write_sample_counts(blueprint_path: str | Path, counts: dict[str, int]) -> None:
    """Atomically persist confirmed task counts in the working Blueprint."""
    path = Path(blueprint_path)
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(counts):
        raise ValueError("Sample counts must contain exactly every retained Blueprint task.")
    total = blueprint.get("benchmark_info", {}).get("sample_count")
    normalized = distribute_sample_counts(list(tasks), total, counts)
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
