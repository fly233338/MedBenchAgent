"""Composable v2 Task Discovery-to-Benchmark pipeline boundary."""

from __future__ import annotations

from typing import Any, Iterable

from medbenchagent.audit.manifest_v2 import build_manifest
from medbenchagent.audit.v2 import audit_task_items
from medbenchagent.generation.v2 import generate_task_items
from medbenchagent.planning.allocation_v2 import allocate_task_leaves
from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.task_discovery import validate_candidate_leaf
from medbenchagent.planning.task_space import AxisAnalysis, TaskLeaf, TaskNode, TaskSpacePlan


def _capability_report(nodes: dict[str, TaskNode], leaves: dict[str, TaskLeaf], items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    groups = defaultdict(lambda: {"items": 0, "leaves": set(), "nodes": set()})
    for item in items:
        leaf = leaves.get(item.get("task_id"))
        if leaf is None:
            continue
        node = nodes.get(leaf.node_id)
        dimensions = node.capability_dimension_ids if node else []
        for dimension in dimensions or ["unassigned"]:
            key = (dimension, leaf.task_paradigm.value, leaf.output_spec.value)
            group = groups[key]
            group["items"] += 1
            group["leaves"].add(leaf.leaf_id)
            group["nodes"].add(leaf.node_id)
    return {
        "schema_version": "capability-report",
        "groups": [
            {"capability_dimension": key[0], "task_paradigm": key[1], "output_spec": key[2],
             "items": value["items"], "leaf_count": len(value["leaves"]), "node_count": len(value["nodes"])}
            for key, value in sorted(groups.items())
        ],
    }


def run_v2_pipeline(
    *,
    dataset: dict[str, Any],
    axis_analysis: AxisAnalysis | dict[str, Any] | None = None,
    node_context: dict[str, Any] | None = None,
    leaves: Iterable[TaskLeaf | dict[str, Any]] = (),
    cases: Iterable[dict[str, Any]],
    requested_count: int,
    nodes: Iterable[TaskNode | dict[str, Any]] | None = None,
    plan: TaskSpacePlan | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the deterministic v2 stages and return all reviewable artifacts."""
    parsed_plan = TaskSpacePlan.model_validate(plan) if plan is not None else None
    analysis = AxisAnalysis.model_validate(axis_analysis or parsed_plan.axis_analysis)
    parsed_nodes = (
        {node.node_id: node for node in (TaskNode.model_validate(value) for value in (nodes or parsed_plan.nodes.values()))}
        if nodes is not None or parsed_plan is not None
        else build_task_nodes(analysis, node_context or {})
    )
    parsed_leaves = []
    raw_evidence = (node_context or {}).get("evidence", [])
    if parsed_plan is not None:
        raw_evidence = parsed_plan.dataset_facts.get("evidence_names", raw_evidence)
    evidence_names = set(raw_evidence) if isinstance(raw_evidence, dict) else {row.get("name") for row in raw_evidence}
    for value in leaves:
        leaf = TaskLeaf.model_validate(value)
        validate_candidate_leaf(leaf, parsed_nodes, resolver_context={"evidence_names": evidence_names})
        parsed_leaves.append(leaf)
    case_list = list(cases)
    from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
    parsed_leaves = [leaf.model_copy(update={"eligible_count": sum(
        not isinstance(resolve(case, ResolverSpec.parse(leaf.resolver)), NotApplicable) for case in case_list
    )}) for leaf in parsed_leaves]
    allocation = allocate_task_leaves(parsed_leaves, requested_count)
    active = [leaf for leaf in parsed_leaves if allocation.get(leaf.leaf_id, 0) > 0]
    items = generate_task_items(case_list, active, dataset.get("dataset_id", "dataset"), allocation=allocation)
    audit = audit_task_items(case_list, items, active)
    manifest = build_manifest(dataset, parsed_nodes.values(), parsed_leaves, allocation,
                              item_count=audit["accepted"], generated_items=audit["accepted_items"])
    shortfall = {leaf.leaf_id: allocation.get(leaf.leaf_id, 0) - sum(item.get("task_id") == leaf.leaf_id for item in audit["accepted_items"])
                 for leaf in parsed_leaves if allocation.get(leaf.leaf_id, 0) > 0}
    leaves_by_id = {leaf.leaf_id: leaf for leaf in parsed_leaves}
    return {"axis_analysis": analysis.model_dump(), "nodes": {key: node.model_dump() for key, node in parsed_nodes.items()},
            "leaves": {leaf.leaf_id: leaf.model_dump() for leaf in parsed_leaves}, "allocation": allocation,
            "items": audit["accepted_items"], "audit": {key: value for key, value in audit.items() if key not in {"accepted_items", "rejected_items"}},
            "shortfall": {key: value for key, value in shortfall.items() if value > 0}, "manifest": manifest,
            "capability_report": _capability_report(parsed_nodes, leaves_by_id, audit["accepted_items"])}

