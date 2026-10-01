"""Manifest construction for v2 benchmark artifacts."""

from __future__ import annotations

from typing import Any, Iterable

from medbenchagent.planning.task_space import TaskLeaf, TaskNode


def build_manifest(dataset: dict[str, Any], nodes: Iterable[TaskNode | dict], leaves: Iterable[TaskLeaf | dict], allocation: dict[str, int], *, item_count: int) -> dict[str, Any]:
    parsed_nodes = {node.node_id: node for node in (TaskNode.model_validate(value) for value in nodes)}
    parsed_leaves = {leaf.leaf_id: leaf for leaf in (TaskLeaf.model_validate(value) for value in leaves)}
    unknown = set(allocation) - set(parsed_leaves)
    if unknown:
        raise ValueError(f"Allocation references unknown Task Leaves: {sorted(unknown)}")
    task_records = {}
    for leaf_id, leaf in parsed_leaves.items():
        if leaf.node_id not in parsed_nodes:
            raise ValueError(f"Task Leaf references unknown Node: {leaf.node_id}")
        task_records[leaf_id] = {
            "node_id": leaf.node_id,
            "capability_category": leaf.capability_category,
            "task_paradigm": leaf.task_paradigm.value,
            "output_spec": leaf.output_spec.value,
            "allocated_count": allocation.get(leaf_id, 0),
            "eligible_count": leaf.eligible_count,
        }
    return {
        "schema_version": "benchmark-manifest.v2",
        "dataset": dataset,
        "nodes": {key: value.model_dump() for key, value in parsed_nodes.items()},
        "tasks": task_records,
        "allocation": allocation,
        "item_count": item_count,
        "metrics": {},
    }

