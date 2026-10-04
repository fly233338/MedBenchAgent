"""Global Allocation stage for completed dataset-local Task Plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.planning.allocation import allocate_task_plans
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.llm import complete_json


class AllocationWeight(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weight: float = Field(ge=0)
    reason: str = Field(min_length=1)
    preference_references: list[str] = Field(default_factory=list)


class AllocationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    datasets: dict[str, AllocationWeight]
    leaves: dict[str, dict[str, AllocationWeight]]


def _weights_from_preferences(context: dict[str, Any], plans: dict[str, TaskPlanState]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    preferences = context.get("allocation_preferences", [])
    if not preferences:
        return {}, {}
    supplied = context.get("allocation_intent")
    if supplied is None:
        supplied = complete_json(
            """Allocate one global benchmark budget across the supplied datasets and Leaf Tasks.
Return JSON with datasets and leaves maps and cover every key exactly once.
Use non-default weights only when supported by an explicit preference_references ID.
Do not use medical importance or invent priorities.""",
            json.dumps({
                "preferences": preferences,
                "datasets": sorted(plans),
                "leaves": {dataset_id: sorted(plan.leaf_tasks) for dataset_id, plan in plans.items()},
            }, ensure_ascii=False),
            AllocationIntent,
        )
    intent = AllocationIntent.model_validate(supplied)
    dataset_ids = set(plans)
    if set(intent.datasets) != dataset_ids or set(intent.leaves) != dataset_ids:
        raise ValueError("Allocation intent must cover every selected dataset")
    preference_ids = {item["id"] for item in preferences}
    dataset_weights, leaf_weights = {}, {}
    for dataset_id, spec in intent.datasets.items():
        refs = set(spec.preference_references)
        if not refs <= preference_ids or (spec.weight != 1.0 and not refs):
            raise ValueError("Dataset allocation weights require current preference references")
        dataset_weights[dataset_id] = spec.weight
    for dataset_id, values in intent.leaves.items():
        if set(values) != set(plans[dataset_id].leaf_tasks):
            raise ValueError(f"Leaf allocation intent must cover every Leaf in {dataset_id}")
        leaf_weights[dataset_id] = {}
        for leaf_id, spec in values.items():
            refs = set(spec.preference_references)
            if not refs <= preference_ids or (spec.weight != 1.0 and not refs):
                raise ValueError("Leaf allocation weights require current preference references")
            leaf_weights[dataset_id][leaf_id] = spec.weight
    return dataset_weights, leaf_weights


def allocate_benchmark(context: dict[str, Any], *, output_path: str | Path | None = None,
                       dataset_weights: dict[str, float] | None = None,
                       leaf_weights: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    selected = list(context.get("selected_dataset_ids", []))
    if not selected:
        raise ValueError("Dataset selection is required before allocation")
    plans = {
        dataset_id: TaskPlanState.model_validate(json.loads(Path(context["dataset_task_plans"][dataset_id]).read_text(encoding="utf-8")))
        for dataset_id in selected
    }
    requested = context.get("benchmark_info", {}).get("sample_count")
    if type(requested) is not int or requested <= 0:
        raise ValueError("benchmark_info.sample_count must be a positive integer")
    if context.get("allocation_preferences"):
        dataset_weights, leaf_weights = _weights_from_preferences(context, plans)
    allocation = allocate_task_plans(
        plans,
        requested,
        dataset_weights=dataset_weights or None,
        leaf_weights=leaf_weights or None,
    )
    destination = Path(output_path or Path(context["operation_directory"]) / "allocation.json")
    atomic_write_json(destination, allocation)
    return result("success", f"Allocated {allocation['allocated_count']} of {requested} requested items",
                  allocation_path=str(destination), allocation=allocation, planning_shortfall=allocation["shortfall"])


def compile_allocation_agent(*, checkpointer=None):
    """Return the global allocation boundary; only explicit preferences use an LLM."""
    return allocate_benchmark
