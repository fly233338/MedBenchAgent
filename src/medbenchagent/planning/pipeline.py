"""Deterministic Scenario → LeafTask → BenchmarkItem pipeline."""

from __future__ import annotations

from typing import Any, Iterable

from medbenchagent.audit.manifest import build_manifest
from medbenchagent.audit.task_items import audit_task_items
from medbenchagent.generation.items import generate_task_items
from medbenchagent.planning.allocation import allocate_task_leaves
from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.task_discovery import validate_candidate_leaf
from medbenchagent.planning.task_space import AxisAnalysis, LeafTask, TaskPlanState, TaskScenario


def _capability_report(scenarios: dict[str, TaskScenario], leaves: dict[str, LeafTask], items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict
    groups = defaultdict(lambda: {"items": 0, "leaves": set(), "scenarios": set()})
    for item in items:
        leaf = leaves.get(item.get("task_id"))
        if leaf is None:
            continue
        scenario = scenarios.get(leaf.scenario_id)
        category = leaf.capability_category.value
        key = (category, leaf.task_paradigm.value, leaf.output_spec.value)
        group = groups[key]
        group["items"] += 1
        group["leaves"].add(leaf.leaf_id)
        group["scenarios"].add(scenario.scenario_id if scenario else leaf.scenario_id)
    return {
        "schema_version": "capability-report",
        "groups": [
            {
                "capability_category": key[0],
                "task_paradigm": key[1],
                "output_spec": key[2],
                "items": value["items"],
                "leaf_task_count": len(value["leaves"]),
                "scenario_count": len(value["scenarios"]),
            }
            for key, value in sorted(groups.items())
        ],
    }


def run_pipeline(
    *,
    dataset: dict[str, Any],
    axis_analysis: AxisAnalysis | dict[str, Any] | None = None,
    node_context: dict[str, Any] | None = None,
    leaves: Iterable[LeafTask | dict[str, Any]] | None = None,
    cases: Iterable[dict[str, Any]],
    requested_count: int,
    seed: int = 0,
    nodes: Iterable[TaskScenario | dict[str, Any]] | None = None,
    plan: TaskPlanState | dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed_plan = TaskPlanState.model_validate(plan) if plan is not None else None
    if parsed_plan is not None:
        scenarios = dict(parsed_plan.scenarios)
        parsed_leaves = list(parsed_plan.leaf_tasks.values())
        analysis = None
    else:
        analysis = AxisAnalysis.model_validate(axis_analysis)
        parsed_leaves = []
        scenarios = (
            {scenario.scenario_id: scenario for scenario in (TaskScenario.model_validate(value) for value in nodes)}
            if nodes is not None
            else build_task_scenarios(analysis, node_context or {})
        )
    if leaves is not None:
        parsed_leaves = []
        for value in leaves:
            leaf = LeafTask.model_validate(value)
            if leaf.scenario_id not in scenarios:
                legacy_id = leaf.scenario_id.replace("node-", "scenario-", 1)
                if legacy_id in scenarios:
                    leaf = leaf.model_copy(update={"scenario_id": legacy_id})
            parsed_leaves.append(leaf)
    case_list = list(cases)
    evidence_names = set((node_context or {}).get("evidence_names", []))
    if not evidence_names and isinstance((node_context or {}).get("evidence"), dict):
        evidence_names.update((node_context or {}).get("evidence", {}).keys())
    if not evidence_names:
        evidence_names.update(
            item.get("name")
            for case in case_list
            for item in case.get("evidence", [])
            if item.get("name")
        )
    for leaf in parsed_leaves:
        validate_candidate_leaf(leaf, scenarios, resolver_context={
            "evidence_names": evidence_names,
            "evidence_catalog": (node_context or {}).get("evidence_catalog", {}),
        })
    from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
    parsed_leaves = [
        leaf.model_copy(update={"eligible_count": sum(
            not isinstance(resolve(case, ResolverSpec.parse(leaf.resolver)), NotApplicable)
            for case in case_list
        )})
        for leaf in parsed_leaves
    ]
    allocation = allocate_task_leaves(parsed_leaves, requested_count)
    active = [leaf for leaf in parsed_leaves if allocation.get(leaf.leaf_id, 0) > 0]
    items = generate_task_items(
        case_list,
        active,
        dataset.get("dataset_id", "dataset"),
        allocation=allocation,
        seed=seed,
    )
    audit = audit_task_items(case_list, items, active)
    manifest = build_manifest(dataset, scenarios.values(), parsed_leaves, allocation,
                              item_count=audit["accepted"], generated_items=audit["accepted_items"])
    shortfall = {
        leaf.leaf_id: allocation.get(leaf.leaf_id, 0)
        - sum(item.get("task_id") == leaf.leaf_id for item in audit["accepted_items"])
        for leaf in parsed_leaves if allocation.get(leaf.leaf_id, 0) > 0
    }
    leaves_by_id = {leaf.leaf_id: leaf for leaf in parsed_leaves}
    return {
        "axis_analysis": analysis.model_dump(mode="json") if analysis else None,
        "scenarios": {key: scenario.model_dump(mode="json") for key, scenario in scenarios.items()},
        "leaves": {leaf.leaf_id: leaf.model_dump(mode="json") for leaf in parsed_leaves},
        "allocation": allocation,
        "items": audit["accepted_items"],
        "audit": {key: value for key, value in audit.items() if key not in {"accepted_items", "rejected_items"}},
        "shortfall": {key: value for key, value in shortfall.items() if value > 0},
        "manifest": manifest,
        "capability_report": _capability_report(scenarios, leaves_by_id, audit["accepted_items"]),
    }

