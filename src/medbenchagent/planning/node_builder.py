"""Deterministic constrained Task Scenario construction for annotation.v5."""

from __future__ import annotations

from itertools import product
from typing import Any

from medbenchagent.planning.task_space import (
    CAPABILITY_BY_OPERATION,
    AxisAnalysis,
    EvidenceSelector,
    TaskAxis,
    TaskScenario,
    stable_scenario_id,
)


class ScenarioConstraintRegistry:
    """Structural feasibility checks; this class performs no medical inference."""

    @staticmethod
    def check(scenario: TaskScenario, context: dict[str, Any]) -> tuple[bool, str | None]:
        evidence = context.get("evidence", {})
        assets = context.get("assets", {})
        relations = context.get("relations", {})
        primary_images = assets.get("primary_image_assets")
        if primary_images is None:
            primary_images = assets.get("image_assets", 0)
        if primary_images < 1 and any(name not in {"text", "numeric"} for name in evidence):
            primary_images = 1
        if scenario.input_structure == "single-image" and primary_images < 1:
            return False, "single-image requires a primary image asset"
        if scenario.input_structure == "multi-image" and assets.get("multi_image_cases", 0) < 1:
            return False, "multi-image requires at least two primary image assets in a Case"
        if scenario.input_structure == "multi-view" and assets.get("related_image_cases", 0) <= 0:
            return False, "multi-view requires related image assets"
        if scenario.input_structure == "volume" and assets.get("volume_assets", 0) <= 0:
            return False, "volume requires a volume asset"
        if scenario.input_structure == "image-text" and not relations.get("paired_text", 0):
            return False, "image-text requires an explicit image/text relation"
        if scenario.input_structure == "paired-study" and not relations.get("paired_study", 0):
            return False, "paired-study requires two linked study objects"
        if scenario.operation == "localize" and not (evidence.get("region", 0) or evidence.get("index", 0)):
            return False, "localize requires explicit region or index evidence"
        if scenario.operation == "quantify" and not (evidence.get("numeric", 0) or context.get("computable_numeric")):
            return False, "quantify requires numeric or computable evidence"
        if scenario.operation == "compare" and relations.get("comparable", 0) < 2:
            return False, "compare requires two linked comparable evidence items"
        if scenario.operation == "align" and relations.get("related_image_groups", relations.get("related", 0)) < 1:
            return False, "align requires explicit related evidence objects"
        if scenario.operation == "describe" and not (
            evidence.get("text", 0) or relations.get("paired_text", 0) or context.get("structured_reference")
        ):
            return False, "describe requires a structured reference"
        return True, None


NodeConstraintRegistry = ScenarioConstraintRegistry


def build_task_scenarios(
    axis_analysis: AxisAnalysis,
    context: dict[str, Any],
    *,
    registry: ScenarioConstraintRegistry | None = None,
    exclusions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, TaskScenario]:
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
            capability_category=CAPABILITY_BY_OPERATION[operation],
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

