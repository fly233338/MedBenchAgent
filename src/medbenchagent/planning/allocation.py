"""Capacity-aware allocation for one dataset compilation."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping
from medbenchagent.planning.sample_counts import allocate_leaf_counts
from medbenchagent.planning.task_space import LeafTask, TaskPlanState


def allocate_task_leaves(leaves: Iterable[LeafTask | dict], requested_count: int,
                         weights: Mapping[str, float] | None = None) -> dict[str, int]:
    if type(requested_count) is not int or requested_count <= 0:
        raise ValueError("requested_count must be a positive integer")
    parsed = [LeafTask.model_validate(leaf) for leaf in leaves]
    capacities = {leaf.leaf_task_id: int(leaf.eligible_count or 0) for leaf in parsed}
    supplied = dict(weights) if weights is not None else {key: 1.0 for key in capacities}
    if set(supplied) != set(capacities):
        raise ValueError("Leaf weights must cover exactly the plan's leaves")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
           not math.isfinite(v) or v < 0 for v in supplied.values()):
        raise ValueError("Leaf weights must be finite nonnegative numbers")
    active = {key: supplied[key] if capacities[key] > 0 else 0.0 for key in supplied}
    if not any(active.values()):
        return {key: 0 for key in capacities}
    return allocate_leaf_counts(active, capacities, requested_count)["counts"]


def allocate_task_plan(dataset_id: str, plan: TaskPlanState | dict, requested_count: int,
                       *, leaf_weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    plan = TaskPlanState.model_validate(plan)
    counts = allocate_task_leaves(plan.leaf_tasks.values(), requested_count, leaf_weights)
    allocated = sum(counts.values())
    return {
        "schema_version": "bir-allocation",
        "dataset_id": dataset_id,
        "status": "complete" if allocated else "no_capacity",
        "requested_count": requested_count, "allocated_count": allocated,
        "shortfall": requested_count - allocated,
        "dataset_counts": {dataset_id: allocated}, "counts": {dataset_id: counts},
        "capacities": {dataset_id: {key: leaf.eligible_count or 0 for key, leaf in plan.leaf_tasks.items()}},
        "weights": dict(leaf_weights or {key: 1.0 for key in counts}),
    }
