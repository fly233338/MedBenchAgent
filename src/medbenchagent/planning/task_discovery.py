"""Deterministic Task Scenario and Leaf Task validation."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.resolver import ResolverSpec, validate_resolver_policy
from medbenchagent.planning.task_space import (
    AxisAnalysis,
    LeafTask,
    TaskPlanState,
    TaskScenario,
)


def validate_candidate_leaf(
    leaf: LeafTask,
    scenarios: dict[str, TaskScenario],
    *,
    resolver_context: dict[str, Any] | None = None,
    registry=None,
) -> LeafTask:
    scenario = scenarios.get(leaf.scenario_id)
    if scenario is None:
        raise ValueError("Leaf Task references an unknown Task Scenario")
    if scenario.exclusion_reason or scenario.status == "excluded":
        raise ValueError("Leaf Task cannot be created under an excluded Task Scenario")
    if leaf.capability_category != scenario.capability_category:
        raise ValueError("Leaf Task Category differs from its Task Scenario")
    spec = ResolverSpec.parse(leaf.resolver)
    registry = registry or (resolver_context or {}).get("registry")
    if registry is not None:
        validate_resolver_policy(spec, registry.resolver_policy)
    if leaf.input != spec.inputs or leaf.output_spec != spec.output_spec:
        raise ValueError("Leaf Task input and OutputSpec must exactly match Resolver")
    contracts = (resolver_context or {}).get("registry").task_contracts if (resolver_context or {}).get("registry") else None
    if contracts:
        allowed = set(contracts.get("paradigm_output_specs", {}).get(leaf.task_paradigm.value, []))
        if leaf.output_spec.value not in allowed:
            raise ValueError("OutputSpec is incompatible with TaskParadigm")
        if leaf.task_paradigm.value == "Report Generation" and any(
            selector.evidence_type not in set(contracts.get("report_generation_input_types", ["asset"]))
            for selector in leaf.input
        ):
            raise ValueError("Report Generation accepts only Registry-approved input types")
    elif leaf.task_paradigm.value == "Report Generation" and any(
        selector.evidence_type != "asset" for selector in leaf.input
    ):
        raise ValueError("Report Generation accepts only assets as input")
    if resolver_context is not None:
        names = resolver_context.get("evidence_names", set())
        missing = [
            selector.role for selector in leaf.input
            if selector.name and selector.name not in names
        ]
        if missing:
            raise ValueError(f"Input Evidence roles cannot be resolved: {missing}")
    return leaf


def leaf_identity(leaf: LeafTask):
    spec = ResolverSpec.parse(leaf.resolver).model_dump(mode="json")
    return (
        leaf.scenario_id,
        leaf.target.strip().casefold(),
        json.dumps(spec, sort_keys=True),
        leaf.task_paradigm.value,
    )


def validate_task_tree(plan: TaskPlanState, *, require_reviews: bool = True, registry=None) -> TaskPlanState:
    if not plan.axis_decisions:
        raise ValueError("Task Plan requires completed Axis decisions")
    active = {key: value for key, value in plan.scenarios.items() if not value.exclusion_reason}
    identities: set[tuple[str, str, str, str]] = set()
    leaf_ids: set[str] = set()
    for leaf_id, leaf in plan.leaf_tasks.items():
        validate_candidate_leaf(
            leaf,
            plan.scenarios,
            resolver_context={"evidence_names": set()},
            registry=registry,
        )
        if leaf_id != leaf.leaf_id:
            raise ValueError(f"Leaf map key does not match leaf_task_id: {leaf_id}")
        identity = leaf_identity(leaf)
        if identity in identities:
            raise ValueError("Duplicate Leaf Task semantics")
        identities.add(identity)
        leaf_ids.add(leaf_id)
    for scenario_id, scenario in active.items():
        attached = [
            leaf_id for leaf_id in leaf_ids
            if plan.leaf_tasks[leaf_id].scenario_id == scenario_id
        ]
        if not attached:
            raise ValueError(f"Executable Task Scenario has no Leaf Task: {scenario_id}")
        if sorted(scenario.leaf_task_ids) != sorted(attached):
            raise ValueError(f"Scenario leaf_task_ids do not match leaves: {scenario_id}")
        if require_reviews:
            review = plan.reviews.get(scenario_id)
            if review is None or review.status != "approved" or set(review.checked_leaf_task_ids) != set(attached):
                raise ValueError(f"Scenario has no complete approval review: {scenario_id}")
        if any((plan.leaf_tasks[key].eligible_count or 0) <= 0 for key in attached):
            raise ValueError(f"Scenario contains a zero-capacity Leaf Task: {scenario_id}")
        if scenario.status != "completed":
            raise ValueError(f"Scenario is not completed: {scenario_id}")
    if set(plan.pending_scenario_ids) or plan.current_scenario_id:
        raise ValueError("Task Plan still has pending or active Scenarios")
    return plan


def compile_task_space_plan(
    *,
    dataset: dict[str, Any],
    axis_analysis: AxisAnalysis,
    node_context: dict[str, Any],
    leaves: Iterable[LeafTask] = (),
    revision: int = 0,
    registry=None,
) -> TaskPlanState:
    scenarios = build_task_scenarios(axis_analysis, node_context, registry=registry)
    accepted: dict[str, LeafTask] = {}
    identities: set[tuple[str, str, str, str]] = set()
    for leaf in leaves:
        validate_candidate_leaf(
            leaf,
            scenarios,
            resolver_context={"evidence_names": node_context.get("evidence_names", [])},
            registry=registry,
        )
        identity = leaf_identity(leaf)
        if identity in identities:
            raise ValueError("Duplicate Leaf Task semantics")
        identities.add(identity)
        if leaf.leaf_id in accepted:
            raise ValueError(f"Duplicate Leaf Task identity: {leaf.leaf_id}")
        accepted[leaf.leaf_id] = leaf
        scenario = scenarios[leaf.scenario_id]
        scenarios[leaf.scenario_id] = scenario.model_copy(update={
            "leaf_task_ids": [*scenario.leaf_task_ids, leaf.leaf_id],
            "status": "completed" if leaf.eligible_count else scenario.status,
        })
    active = [key for key, value in scenarios.items() if not value.exclusion_reason]
    completed = [key for key in active if scenarios[key].leaf_task_ids]
    pending = [key for key in active if key not in completed]
    return TaskPlanState(
        dataset=dataset,
        axis_decisions={
            "input_structure": axis_analysis.input_structure,
            "operation": axis_analysis.operation,
            "clinical_target": axis_analysis.clinical_target,
        },
        scenarios=scenarios,
        leaf_tasks=accepted,
        phase="leaves",
        current_scenario_id=pending[0] if pending else None,
        pending_scenario_ids=pending,
        completed_scenario_ids=completed,
        revision=revision,
    )


def node_statistics(nodes: dict[str, TaskScenario]) -> dict[str, Any]:
    return {
        "total": len(nodes),
        "excluded": sum(bool(node.exclusion_reason) for node in nodes.values()),
        "with_leaves": sum(bool(node.leaf_task_ids) for node in nodes.values()),
        "by_capability_category": dict(Counter(node.capability_category for node in nodes.values())),
    }
