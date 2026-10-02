"""Deterministic Task Tree validation and orchestration helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.resolver import ResolverSpec
from medbenchagent.planning.task_space import AxisAnalysis, TaskLeaf, TaskNode, TaskSpacePlan


def validate_candidate_leaf(
    leaf: TaskLeaf,
    nodes: dict[str, TaskNode],
    *,
    resolver_context: dict[str, Any] | None = None,
) -> TaskLeaf:
    """Validate the structural portion of a Leaf before case compilation."""
    node = nodes.get(leaf.node_id)
    if node is None:
        raise ValueError("Task Leaf references an unknown Task Node")
    if node.exclusion_reason or node.status == "excluded":
        raise ValueError("Task Leaf cannot be created under an excluded Task Node")
    if leaf.capability_category != node.capability_category:
        raise ValueError("Task Leaf operation family differs from its Task Node")
    if leaf.target.strip().casefold() != node.clinical_target.strip().casefold():
        raise ValueError("Task Leaf target must match its Task Node clinical target")
    spec = ResolverSpec.parse(leaf.resolver)
    if leaf.input != spec.inputs or leaf.output_spec != spec.output_spec:
        raise ValueError("Task Leaf input and OutputSpec must exactly match Resolver")
    if leaf.task_paradigm.value == "Report Generation" and any(s.evidence_type != "asset" for s in leaf.input):
        raise ValueError("Report Generation accepts only assets as input")
    if resolver_context is not None:
        names = resolver_context.get("evidence_names", set())
        missing = [selector.role for selector in leaf.input if selector.name and selector.name not in names]
        if missing:
            raise ValueError(f"Input Evidence roles cannot be resolved: {missing}")
    return leaf


def leaf_identity(leaf: TaskLeaf):
    import json

    spec = ResolverSpec.parse(leaf.resolver).model_dump(mode="json")
    return (leaf.node_id, leaf.target.strip().casefold(), json.dumps(spec, sort_keys=True), leaf.task_paradigm.value)


def validate_task_tree(
    plan: TaskSpacePlan,
    *,
    require_capabilities: bool = False,
    require_reviews: bool = True,
) -> TaskSpacePlan:
    """Validate the complete persisted draft before it becomes a Plan artifact."""
    if plan.axis_analysis is None:
        raise ValueError("Task Tree requires completed Axis analysis")
    active_nodes = {key: node for key, node in plan.nodes.items() if not node.exclusion_reason}
    leaf_ids: set[str] = set()
    identities: set[tuple[str, str, str, str]] = set()
    for leaf_id, leaf in plan.leaves.items():
        evidence_names = set(plan.dataset_facts.get("evidence_names", []))
        evidence_names.update(
            selector.name for selector in leaf.input
            if selector.name and selector.evidence_type == "asset"
        )
        validate_candidate_leaf(
            leaf,
            plan.nodes,
            resolver_context={"evidence_names": evidence_names},
        )
        if leaf_id != leaf.leaf_id:
            raise ValueError(f"Leaf map key does not match leaf_id: {leaf_id}")
        identity = leaf_identity(leaf)
        if identity in identities:
            raise ValueError("Duplicate Task Leaf semantics")
        identities.add(identity)
        leaf_ids.add(leaf_id)
    for node_id, node in active_nodes.items():
        attached = [leaf_id for leaf_id in leaf_ids if plan.leaves[leaf_id].node_id == node_id]
        if not attached:
            raise ValueError(f"Executable Task Node has no Task Leaf: {node_id}")
        if sorted(node.leaf_ids) != sorted(attached):
            raise ValueError(f"Task Node leaf_ids do not match leaves: {node_id}")
        if require_reviews:
            review = plan.reviews.get(node_id)
            if review is None or review.status != "approved" or set(review.checked_leaf_ids) != set(attached):
                raise ValueError(f"Task Node has no complete approval review: {node_id}")
        if any((plan.leaves[leaf_id].eligible_count or 0) <= 0 for leaf_id in attached):
            raise ValueError(f"Task Node contains a zero-capacity Task Leaf: {node_id}")
        if node.status != "completed":
            raise ValueError(f"Task Node is not completed: {node_id}")
        if require_capabilities and not plan.capability_assignments.get(node_id):
            raise ValueError(f"Task Node has no Blueprint capability assignment: {node_id}")
    if set(plan.pending_node_ids) or plan.current_node_id:
        raise ValueError("Task Tree still has pending or active Nodes")
    if require_capabilities and set(plan.capability_assignments) != set(active_nodes):
        raise ValueError("Capability assignments do not cover all executable Nodes")
    return plan


def compile_task_space_plan(
    *,
    dataset: dict[str, Any],
    axis_analysis: AxisAnalysis,
    node_context: dict[str, Any],
    leaves: Iterable[TaskLeaf] = (),
    revision: int = 0,
) -> TaskSpacePlan:
    """Build a deterministic draft from legacy callers and validate Leaf links."""
    nodes = build_task_nodes(axis_analysis, node_context)
    accepted: dict[str, TaskLeaf] = {}
    identities: set[tuple[str, str, str, str]] = set()
    for leaf in leaves:
        validate_candidate_leaf(
            leaf,
            nodes,
            resolver_context={"evidence_names": node_context.get("evidence_names", [])},
        )
        identity = leaf_identity(leaf)
        if identity in identities:
            raise ValueError("Duplicate Task Leaf semantics")
        identities.add(identity)
        if leaf.leaf_id in accepted:
            raise ValueError(f"Duplicate Task Leaf identity: {leaf.leaf_id}")
        accepted[leaf.leaf_id] = leaf
        node = nodes[leaf.node_id]
        nodes[leaf.node_id] = node.model_copy(update={
            "leaf_ids": [*node.leaf_ids, leaf.leaf_id],
            "status": "completed" if leaf.eligible_count else node.status,
        })
    active = [key for key, node in nodes.items() if not node.exclusion_reason]
    completed = [key for key in active if nodes[key].leaf_ids]
    pending = [key for key in active if key not in completed]
    return TaskSpacePlan(
        schema_version="task-space-plan",
        dataset=dataset,
        axis_analysis=axis_analysis,
        nodes=nodes,
        leaves=accepted,
        dataset_facts=node_context,
        phase="leaves",
        current_node_id=pending[0] if pending else None,
        pending_node_ids=pending,
        completed_node_ids=completed,
        revision=revision,
    )


def node_statistics(nodes: dict[str, TaskNode]) -> dict[str, Any]:
    return {
        "total": len(nodes),
        "excluded": sum(bool(node.exclusion_reason) for node in nodes.values()),
        "with_leaves": sum(bool(node.leaf_ids) for node in nodes.values()),
        "by_operation_family": dict(Counter(node.operation_family for node in nodes.values())),
    }
