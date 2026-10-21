"""Capacity-aware allocation for one dataset compilation."""
from __future__ import annotations

import math
import json
from pydantic import BaseModel, ConfigDict, Field
from medbenchagent.planning.bir import BIRStore, business_dependencies
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.llm import complete_json
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
        "counts": counts,
        "capacities": {key: leaf.eligible_count or 0 for key, leaf in plan.leaf_tasks.items()},
        "weights": dict(leaf_weights or {key: 1.0 for key in counts}),
    }


class AllocationWeight(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weight: float = Field(ge=0, allow_inf_nan=False)
    reason: str = Field(min_length=1)
    preference_references: list[str] = Field(default_factory=list)


class AllocationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    leaves: dict[str, AllocationWeight]


def allocate_benchmark(context: dict[str, Any]) -> dict[str, Any]:
    store = BIRStore(context["bir_path"])
    dataset_id = store.load("dataset_binding.json")["access"]["dataset_id"]
    leaf_weights = None
    brief = store.load("benchmark_info.json")
    plan = TaskPlanState.model_validate(store.load("task_plan.json"))
    if plan.phase != "complete":
        raise ValueError("Allocation requires a complete Task Plan")
    preferences = brief.get("allocation_preferences", [])
    if preferences and plan.leaf_tasks:
        supplied = complete_json(
            "Allocate a single dataset's sample budget across its Leaf Tasks. "
            "Return leaves weights covering every supplied key. Non-default weights require "
            "explicit preference_references IDs. Never invent clinical priorities.",
            json.dumps({"preferences": preferences, "leaves": {
                key: {"name": leaf.name, "target": leaf.target, "eligible_count": leaf.eligible_count}
                for key, leaf in plan.leaf_tasks.items()}}, ensure_ascii=False),
            AllocationIntent,
        )
        intent = AllocationIntent.model_validate(supplied)
        if set(intent.leaves) != set(plan.leaf_tasks):
            raise ValueError("Allocation intent must cover every Leaf")
        known = {row["id"] for row in preferences}
        for spec in intent.leaves.values():
            refs = set(spec.preference_references)
            if not refs <= known or (spec.weight != 1.0 and not refs):
                raise ValueError("Allocation weights require current preference references")
        leaf_weights = {key: spec.weight for key, spec in intent.leaves.items()}
    allocation = allocate_task_plan(dataset_id, plan, brief["sample_count"], leaf_weights=leaf_weights)
    store.update("allocation.json", allocation, run_id=context.get("run_id", "unknown"), actor="allocation",
                 dependencies=business_dependencies(context, "benchmark_info.json", "dataset_binding.json",
                                                    "task_plan.json"))
    return result("success", f"Allocated {allocation['allocated_count']} of {brief['sample_count']} items",
                  data=allocation)
