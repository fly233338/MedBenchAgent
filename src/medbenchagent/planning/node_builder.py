"""Deterministic constrained Task Scenario construction for annotation.v5."""

from __future__ import annotations

from itertools import product
from typing import Any

from medbenchagent.planning.task_space import (
    AxisAnalysis,
    CapabilityCategory,
    EvidenceSelector,
    TaskAxis,
    TaskScenario,
    stable_scenario_id,
)
from medbenchagent.planning.bir import RegistrySnapshot


class ScenarioConstraintRegistry:
    """Structural feasibility checks; this class performs no medical inference."""

    def __init__(self, registry: RegistrySnapshot | None = None):
        if registry is None:
            from pathlib import Path
            registry = RegistrySnapshot.from_directory(
                Path(__file__).resolve().parents[3] / "config" / "bir_template" / "registry"
            )
        self.registry = registry

    @staticmethod
    def _value(context: dict[str, Any], path: str):
        value: Any = context
        for part in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    @classmethod
    def _predicate(cls, context: dict[str, Any], predicate: dict[str, Any]) -> bool:
        value = cls._value(context, predicate.get("path", ""))
        if "eq" in predicate:
            return value == predicate["eq"]
        if "gte" in predicate:
            try:
                return value is not None and value >= predicate["gte"]
            except TypeError:
                return False
        if "gt" in predicate:
            try:
                return value is not None and value > predicate["gt"]
            except TypeError:
                return False
        if "exists" in predicate:
            return (value is not None) is bool(predicate["exists"])
        return False

    def check(self, scenario: TaskScenario, context: dict[str, Any]) -> tuple[bool, str | None]:
        context = dict(context)
        assets = dict(context.get("assets", {}))
        if "primary_image_assets" not in assets:
            assets["primary_image_assets"] = assets.get("image_assets", 0)
        if assets.get("primary_image_assets", 0) < 1 and any(
            name not in {"text", "numeric"} for name in context.get("evidence", {})
        ):
            assets["primary_image_assets"] = 1
        context["assets"] = assets
        for rule in self.registry.scenario_constraints.get("rules", []):
            when = rule.get("when", {})
            if any(getattr(scenario, axis, None) != expected for axis, expected in when.items()):
                continue
            required = rule.get("requires", [])
            if required and not all(self._predicate(context, item) for item in required):
                return False, rule.get("message", rule.get("id", "constraint failed"))
            alternatives = rule.get("requires_any", [])
            if alternatives and not any(self._predicate(context, item) for item in alternatives):
                return False, rule.get("message", rule.get("id", "constraint failed"))
        return True, None


NodeConstraintRegistry = ScenarioConstraintRegistry


def build_task_scenarios(
    axis_analysis: AxisAnalysis,
    context: dict[str, Any],
    *,
    registry: ScenarioConstraintRegistry | None = None,
    exclusions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, TaskScenario]:
    if isinstance(registry, RegistrySnapshot):
        registry = ScenarioConstraintRegistry(registry)
    registry = registry or ScenarioConstraintRegistry()
    supported = {
        TaskAxis.INPUT_STRUCTURE: [
            item.value for item in axis_analysis.input_structure if item.supported
        ],
        TaskAxis.OPERATION: [
            item.value for item in axis_analysis.operation if item.supported
        ],
        TaskAxis.CLINICAL_TARGET: [
            item.value for item in axis_analysis.clinical_target if item.supported
        ],
    }
    for axis in (TaskAxis.INPUT_STRUCTURE, TaskAxis.OPERATION, TaskAxis.CLINICAL_TARGET):
        allowed = set(registry.registry.axis_values(axis.value))
        if not set(supported[axis]) <= allowed:
            raise ValueError(f"Axis analysis contains values outside the BIR Registry: {axis.value}")
    scenarios: dict[str, TaskScenario] = {}
    for input_structure, operation, clinical_target in product(
        supported[TaskAxis.INPUT_STRUCTURE],
        supported[TaskAxis.OPERATION],
        supported[TaskAxis.CLINICAL_TARGET],
    ):
        mapping = [
            EvidenceSelector.model_validate(value)
            for value in context.get("annotation_mapping", [])
        ]
        scenario = TaskScenario(
            scenario_id=stable_scenario_id(input_structure, operation, clinical_target),
            input_structure=input_structure,
            operation=operation,
            clinical_target=clinical_target,
            capability_category=CapabilityCategory(registry.registry.capability_by_operation[operation]),
            annotation_mapping=mapping,
            evidence_stats=context.get("evidence", {}),
        )
        feasible, reason = registry.check(scenario, context)
        case_facts = context.get("case_facts", [])
        feasible_count = None
        if case_facts:
            feasible_count = sum(registry.check(scenario, facts)[0] for facts in case_facts)
            if feasible and feasible_count == 0:
                feasible = False
                reason = "no Case satisfies the Scenario structural constraints"
        scenario = scenario.model_copy(update={"feasible_case_count": feasible_count})
        if not feasible:
            if exclusions is not None:
                exclusions[scenario.scenario_id] = {
                    "scenario": scenario.model_dump(mode="json"),
                    "reason": reason,
                }
            scenario = scenario.model_copy(
                update={"exclusion_reason": reason, "status": "excluded"}
            )
        scenarios[scenario.scenario_id] = scenario
    return scenarios


def build_task_nodes(axis_analysis: AxisAnalysis, context: dict[str, Any], *, registry=None, exclusions=None):
    """Deprecated in-process alias; new plans persist Task Scenarios."""
    return build_task_scenarios(axis_analysis, context, registry=registry, exclusions=exclusions)

