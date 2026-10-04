"""Manifest construction for the benchmark artifacts."""

from __future__ import annotations

from typing import Any, Iterable

from medbenchagent.planning.task_space import LeafTask, TaskScenario


def build_manifest(
    dataset: dict[str, Any],
    scenarios: Iterable[TaskScenario | dict],
    leaves: Iterable[LeafTask | dict],
    allocation: dict[str, int],
    *,
    item_count: int,
    generated_items: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    parsed_scenarios = {}
    for raw in scenarios:
        value = dict(raw) if isinstance(raw, dict) else raw.model_dump(mode="json")
        if "capability_category" not in value:
            value["capability_category"] = "Recognition"
        parsed = TaskScenario.model_validate(value)
        parsed_scenarios[parsed.scenario_id] = parsed
    parsed_leaves = {
        leaf.leaf_task_id: leaf
        for leaf in (LeafTask.model_validate(value) for value in leaves)
    }
    unknown = set(allocation) - set(parsed_leaves)
    if unknown:
        raise ValueError(f"Allocation references unknown Leaf Tasks: {sorted(unknown)}")
    generated_counts: dict[str, int] = {}
    for item in generated_items:
        generated_counts[item.get("task_id")] = generated_counts.get(item.get("task_id"), 0) + 1
    task_records = {}
    for leaf_id, leaf in parsed_leaves.items():
        scenario = parsed_scenarios.get(leaf.scenario_id)
        if scenario is None:
            raise ValueError(f"Leaf Task references unknown Scenario: {leaf.scenario_id}")
        task_records[leaf_id] = {
            "scenario_id": leaf.scenario_id,
            "capability_category": leaf.capability_category.value,
            "task_paradigm": leaf.task_paradigm.value,
            "output_spec": leaf.output_spec.value,
            "allocated_count": allocation.get(leaf_id, 0),
            "eligible_count": leaf.eligible_count,
            "generated_count": generated_counts.get(leaf_id, 0),
        }
    return {
        "schema_version": "benchmark-manifest",
        "dataset": dataset,
        "task_scenarios": {key: value.model_dump(mode="json") for key, value in parsed_scenarios.items()},
        "leaf_tasks": {key: value.model_dump(mode="json") for key, value in parsed_leaves.items()},
        "tasks": task_records,
        "allocation": allocation,
        "item_count": item_count,
        "generated_count": item_count,
    }

