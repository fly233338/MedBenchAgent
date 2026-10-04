"""Deterministic capacity-aware allocation for one or more Task Plans."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from medbenchagent.planning.sample_counts import allocate_leaf_counts
from medbenchagent.planning.task_space import LeafTask, TaskLeaf, TaskPlanState


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


def _plan_leaves(plan: TaskPlanState | Mapping[str, Any]) -> dict[str, LeafTask]:
    parsed = plan if isinstance(plan, TaskPlanState) else TaskPlanState.model_validate(plan)
    return dict(parsed.leaf_tasks)


def _validate_weights(weights: Mapping[str, float] | None, keys: set[str], label: str) -> dict[str, float]:
    if weights is None:
        return {key: 1.0 for key in sorted(keys)}
    if set(weights) != keys:
        raise ValueError(f"{label} weights must cover exactly: {sorted(keys)}")
    normalized = {}
    for key, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{label} weights must be nonnegative numbers")
        normalized[key] = float(value)
    if not any(normalized.values()):
        raise ValueError(f"{label} weights must contain a positive value")
    return normalized


def allocate_task_plans(
    plans: Mapping[str, TaskPlanState | Mapping[str, Any]],
    requested_count: int,
    *,
    dataset_weights: Mapping[str, float] | None = None,
    leaf_weights: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Allocate a global budget dataset first, then Leaf within each dataset.

    ``requested_count`` is global across all selected datasets.  Capacities are
    the exact ``eligible_count`` values persisted by Task Planning.  The
    allocator never creates a count greater than a Leaf's capacity and reports
    any final shortfall explicitly.
    """
    if type(requested_count) is not int or requested_count <= 0:
        raise ValueError("requested_count must be a positive integer")
    if not plans:
        raise ValueError("At least one Task Plan is required")
    parsed = {str(dataset_id): _plan_leaves(plan) for dataset_id, plan in plans.items()}
    capacities = {
        dataset_id: {leaf_id: int(leaf.eligible_count or 0) for leaf_id, leaf in leaves.items()}
        for dataset_id, leaves in parsed.items()
    }
    active_datasets = {dataset_id for dataset_id, leaves in capacities.items() if any(leaves.values())}
    if not active_datasets:
        return {
            "schema_version": "bir-allocation",
            "status": "no_capacity",
            "requested_count": requested_count,
            "allocated_count": 0,
            "shortfall": requested_count,
            "dataset_counts": {dataset_id: 0 for dataset_id in sorted(capacities)},
            "counts": {dataset_id: {leaf_id: 0 for leaf_id in sorted(leaves)} for dataset_id, leaves in capacities.items()},
            "capacities": capacities,
            "weights": {"datasets": {}, "leaves": {}},
        }

    dataset_weight_map = _validate_weights(dataset_weights, set(plans), "Dataset")
    # Zero-capacity datasets remain in the published allocation, but cannot
    # receive quota even when a caller supplies a positive weight.
    dataset_weight_map = {
        dataset_id: weight if dataset_id in active_datasets else 0.0
        for dataset_id, weight in dataset_weight_map.items()
    }
    if not any(dataset_weight_map.values()):
        raise ValueError("No selected dataset has positive allocation capacity")
    dataset_capacity = {dataset_id: sum(capacities[dataset_id].values()) for dataset_id in capacities}
    first = allocate_leaf_counts(dataset_weight_map, dataset_capacity, requested_count)

    counts = {dataset_id: {leaf_id: 0 for leaf_id in sorted(leaves)} for dataset_id, leaves in capacities.items()}
    leaf_weight_records: dict[str, dict[str, float]] = {}
    for dataset_id in sorted(capacities):
        leaves = capacities[dataset_id]
        if not first["counts"].get(dataset_id):
            leaf_weight_records[dataset_id] = {}
            continue
        provided = (leaf_weights or {}).get(dataset_id)
        weights = _validate_weights(provided, set(leaves), f"Leaf {dataset_id}")
        weights = {leaf_id: weight if leaves[leaf_id] > 0 else 0.0 for leaf_id, weight in weights.items()}
        if not any(weights.values()):
            leaf_weight_records[dataset_id] = {}
            continue
        leaf_weight_records[dataset_id] = weights
        result = allocate_leaf_counts(weights, leaves, first["counts"][dataset_id])
        counts[dataset_id].update(result["counts"])

    allocated = sum(sum(values.values()) for values in counts.values())
    return {
        "schema_version": "bir-allocation",
        "status": "complete" if allocated else "no_capacity",
        "requested_count": requested_count,
        "allocated_count": allocated,
        "shortfall": requested_count - allocated,
        "dataset_counts": {dataset_id: sum(values.values()) for dataset_id, values in counts.items()},
        "counts": counts,
        "capacities": capacities,
        "weights": {"datasets": dataset_weight_map, "leaves": leaf_weight_records},
    }

