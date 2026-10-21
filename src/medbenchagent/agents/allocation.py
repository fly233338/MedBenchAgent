"""Dataset-local allocation; only explicit preferences require semantic interpretation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from medbenchagent.planning.allocation import allocate_task_plan
from medbenchagent.planning.bir import BIRStore, business_dependencies
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.llm import complete_json
from medbenchagent.runtime.run_artifacts import atomic_write_json


class AllocationWeight(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weight: float = Field(ge=0, allow_inf_nan=False)
    reason: str = Field(min_length=1)
    preference_references: list[str] = Field(default_factory=list)


class AllocationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    leaves: dict[str, AllocationWeight]


def allocate_benchmark(context: dict[str, Any], *, output_path: str | Path | None = None,
                       leaf_weights: dict[str, float] | None = None) -> dict[str, Any]:
    store = BIRStore(context["bir_path"])
    selected = list(store.load("dataset_bindings.json")["datasets"])
    if len(selected) != 1:
        raise ValueError("Allocation requires exactly one dataset")
    dataset_id = selected[0]
    brief = store.load("benchmark_info.json")
    plan = TaskPlanState.model_validate(store.load(f"datasets/{dataset_id}/planning/task_plan.json"))
    if plan.phase != "complete":
        raise ValueError("Allocation requires a complete Task Plan")
    preferences = brief.get("allocation_preferences", [])
    if preferences and plan.leaf_tasks:
        supplied = context.get("allocation_intent")
        if supplied is None:
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
    destination = Path(output_path or Path(context["operation_directory"]) / "allocation.json")
    atomic_write_json(destination, allocation)
    store.update("allocation.json", allocation, run_id=context.get("run_id", "unknown"), actor="allocation",
                 dependencies=business_dependencies(context, "benchmark_info.json", "dataset_bindings.json",
                                                    f"datasets/{dataset_id}/planning/task_plan.json"))
    return result("success", f"Allocated {allocation['allocated_count']} of {brief['sample_count']} items",
                  allocation_path=str(store.root / "allocation.json"), allocation=allocation,
                  planning_shortfall=allocation["shortfall"])
