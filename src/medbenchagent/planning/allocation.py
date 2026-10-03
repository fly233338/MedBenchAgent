"""Deterministic capacity-aware allocation for Task Leaves."""

from __future__ import annotations

from typing import Iterable

from medbenchagent.planning.task_space import TaskLeaf


def allocate_task_leaves(leaves: Iterable[TaskLeaf | dict], requested_count: int) -> dict[str, int]:
    if type(requested_count) is not int or requested_count <= 0:
        raise ValueError("requested_count must be a positive integer")
    parsed = [TaskLeaf.model_validate(leaf) for leaf in leaves]
    capacities = {leaf.leaf_id: leaf.eligible_count or 0 for leaf in parsed}
    active = [key for key, capacity in capacities.items() if capacity > 0]
    if not active:
        return {key: 0 for key in capacities}
    total = min(requested_count, sum(capacities.values()))
    base, remainder = divmod(total, len(active))
    allocation = {key: 0 for key in capacities}
    for index, key in enumerate(sorted(active)):
        allocation[key] = min(capacities[key], base + (index < remainder))
    remaining = total - sum(allocation.values())
    for key in sorted(active):
        if remaining <= 0:
            break
        extra = min(capacities[key] - allocation[key], remaining)
        allocation[key] += extra
        remaining -= extra
    return allocation

