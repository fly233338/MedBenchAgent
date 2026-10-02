"""Manifest construction for v2 benchmark artifacts."""

from __future__ import annotations

from typing import Any, Iterable

from medbenchagent.planning.task_space import TaskLeaf, TaskNode


def build_manifest(dataset: dict[str, Any], nodes: Iterable[TaskNode | dict], leaves: Iterable[TaskLeaf | dict], allocation: dict[str, int], *, item_count: int, generated_items: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    parsed_nodes = {node.node_id: node for node in (TaskNode.model_validate(value) for value in nodes)}
    parsed_leaves = {leaf.leaf_id: leaf for leaf in (TaskLeaf.model_validate(value) for value in leaves)}
    unknown = set(allocation) - set(parsed_leaves)
    if unknown:
        raise ValueError(f"Allocation references unknown Task Leaves: {sorted(unknown)}")
    generated_counts = {}
    for item in generated_items:
        generated_counts[item.get("task_id")] = generated_counts.get(item.get("task_id"), 0) + 1
    task_records = {}
    for leaf_id, leaf in parsed_leaves.items():
        if leaf.node_id not in parsed_nodes:
            raise ValueError(f"Task Leaf references unknown Node: {leaf.node_id}")
        task_records[leaf_id] = {
            "node_id": leaf.node_id,
            "capability_category": leaf.capability_category,
            "capability_dimension_ids": parsed_nodes[leaf.node_id].capability_dimension_ids,
            "task_paradigm": leaf.task_paradigm.value,
            "output_spec": leaf.output_spec.value,
            "allocated_count": allocation.get(leaf_id, 0),
            "eligible_count": leaf.eligible_count,
            "generated_count": generated_counts.get(leaf_id, 0),
        }
    for task_id, count in allocation.items():
        if task_id in task_records:
            task_records[task_id]["allocated_count"] = count
    return {
        "schema_version": "benchmark-manifest.v2",
        "dataset": dataset,
        "nodes": {key: value.model_dump() for key, value in parsed_nodes.items()},
        "tasks": task_records,
        "allocation": allocation,
        "item_count": item_count,
        "generated_count": item_count,
    }

