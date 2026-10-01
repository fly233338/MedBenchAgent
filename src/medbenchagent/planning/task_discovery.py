"""Pure orchestration helpers for the v2 Task Discovery pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.resolver import ResolverSpec
from medbenchagent.planning.task_space import AxisAnalysis, TaskLeaf, TaskNode, TaskSpacePlan


def validate_candidate_leaf(leaf: TaskLeaf, nodes: dict[str, TaskNode], *, resolver_context: dict[str, Any] | None = None) -> TaskLeaf:
    """Validate the structural portion of a candidate before case-level compilation."""
    node = nodes.get(leaf.node_id)
    if node is None:
        raise ValueError("Task Leaf references an unknown Task Node")
    if node.exclusion_reason:
        raise ValueError("Task Leaf cannot be created under an excluded Task Node")
    if leaf.capability_category != node.capability_category:
        raise ValueError("Task Leaf capability category differs from its Task Node")
    spec = ResolverSpec.parse(leaf.resolver)
    if leaf.input != spec.inputs or leaf.output_spec != spec.output_spec:
        raise ValueError("Task Leaf input and OutputSpec must exactly match Resolver")
    if leaf.task_paradigm.value == "Report Generation" and any(s.evidence_type != "asset" for s in leaf.input):
        raise ValueError("Report Generation accepts only assets as input")
    if resolver_context is not None:
        missing = [selector.role for selector in leaf.input if selector.name and selector.name not in resolver_context.get("evidence_names", set())]
        if missing:
            raise ValueError(f"Input Evidence roles cannot be resolved: {missing}")
    return leaf


def compile_task_space_plan(
    *, dataset: dict[str, Any], axis_analysis: AxisAnalysis, node_context: dict[str, Any], leaves: Iterable[TaskLeaf] = (), revision: int = 0,
) -> TaskSpacePlan:
    """Build and validate a complete deterministic task-space plan."""
    nodes = build_task_nodes(axis_analysis, node_context)
    accepted: dict[str, TaskLeaf] = {}
    identities: set[tuple[str, str, str, str]] = set()
    for leaf in leaves:
        validate_candidate_leaf(leaf, nodes, resolver_context={"evidence_names": node_context.get("evidence_names", [])})
        identity = leaf_identity(leaf)
        if identity in identities:
            raise ValueError("Duplicate Task Leaf semantics")
        identities.add(identity)
        if leaf.leaf_id in accepted:
            raise ValueError(f"Duplicate Task Leaf identity: {leaf.leaf_id}")
        accepted[leaf.leaf_id] = leaf
        nodes[leaf.node_id] = nodes[leaf.node_id].model_copy(update={"leaf_ids": [*nodes[leaf.node_id].leaf_ids, leaf.leaf_id]})
    return TaskSpacePlan(dataset=dataset, axis_analysis=axis_analysis, nodes=nodes, leaves=accepted, revision=revision)


def leaf_identity(leaf):
    import json
    spec = ResolverSpec.parse(leaf.resolver).model_dump(mode="json")
    return (leaf.node_id, leaf.target.strip().casefold(), json.dumps(spec, sort_keys=True), leaf.task_paradigm.value)


def node_statistics(nodes: dict[str, TaskNode]) -> dict[str, Any]:
    return {
        "total": len(nodes),
        "excluded": sum(bool(node.exclusion_reason) for node in nodes.values()),
        "with_leaves": sum(bool(node.leaf_ids) for node in nodes.values()),
        "by_capability": dict(Counter(node.capability_category for node in nodes.values())),
    }

