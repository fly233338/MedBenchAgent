"""Composable v2 Task Discovery-to-Benchmark pipeline boundary."""

from __future__ import annotations

from typing import Any, Iterable

from medbenchagent.audit.manifest_v2 import build_manifest
from medbenchagent.audit.v2 import audit_task_items
from medbenchagent.generation.v2 import generate_task_items
from medbenchagent.planning.allocation_v2 import allocate_task_leaves
from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.task_discovery import validate_candidate_leaf
from medbenchagent.planning.task_space import AxisAnalysis, TaskLeaf


def run_v2_pipeline(*, dataset: dict[str, Any], axis_analysis: AxisAnalysis | dict[str, Any], node_context: dict[str, Any], leaves: Iterable[TaskLeaf | dict[str, Any]], cases: Iterable[dict[str, Any]], requested_count: int) -> dict[str, Any]:
    """Run the deterministic v2 stages and return all reviewable artifacts."""
    analysis = AxisAnalysis.model_validate(axis_analysis)
    nodes = build_task_nodes(analysis, node_context)
    parsed_leaves = []
    raw_evidence = node_context.get("evidence", [])
    evidence_names = set(raw_evidence) if isinstance(raw_evidence, dict) else {row.get("name") for row in raw_evidence}
    for value in leaves:
        leaf = TaskLeaf.model_validate(value)
        validate_candidate_leaf(leaf, nodes, resolver_context={"evidence_names": evidence_names})
        parsed_leaves.append(leaf)
    case_list = list(cases)
    from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
    parsed_leaves = [leaf.model_copy(update={"eligible_count": sum(
        not isinstance(resolve(case, ResolverSpec.parse(leaf.resolver)), NotApplicable) for case in case_list
    )}) for leaf in parsed_leaves]
    allocation = allocate_task_leaves(parsed_leaves, requested_count)
    active = [leaf for leaf in parsed_leaves if allocation.get(leaf.leaf_id, 0) > 0]
    items = generate_task_items(case_list, active, dataset.get("dataset_id", "dataset"))
    audit = audit_task_items(case_list, items, active)
    manifest = build_manifest(dataset, nodes.values(), parsed_leaves, allocation, item_count=audit["accepted"])
    return {"axis_analysis": analysis.model_dump(), "nodes": {key: node.model_dump() for key, node in nodes.items()},
            "leaves": {leaf.leaf_id: leaf.model_dump() for leaf in parsed_leaves}, "allocation": allocation,
            "items": audit["accepted_items"], "audit": {key: value for key, value in audit.items() if key not in {"accepted_items", "rejected_items"}},
            "manifest": manifest}

