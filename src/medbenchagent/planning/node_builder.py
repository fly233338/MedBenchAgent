"""Deterministic constrained Task Node construction for annotation.v5."""

from __future__ import annotations

from itertools import product
from typing import Any

from medbenchagent.planning.task_space import (
    CAPABILITY_BY_OPERATION, AxisAnalysis, TaskAxis, TaskNode,
    stable_node_id,
)


class NodeConstraintRegistry:
    """Structural feasibility checks; this class deliberately performs no medical inference."""

    @staticmethod
    def check(node: TaskNode, context: dict[str, Any]) -> tuple[bool, str | None]:
        evidence = context.get("evidence", {})
        assets = context.get("assets", {})
        relations = context.get("relations", {})
        if node.input_structure == "single-image" and "image_assets" in assets and assets.get("image_assets", 0) < 1:
            return False, "single-image requires an image asset"
        if node.input_structure == "multi-image" and "multi_image_cases" in assets and assets.get("multi_image_cases", 0) < 1:
            return False, "multi-image requires multiple assets in a Case"
        if node.input_structure == "multi-view" and assets.get("related_image_cases", 0) <= 0:
            return False, "multi-view requires related image assets"
        if node.input_structure == "volume" and assets.get("volume_assets", 0) <= 0:
            return False, "volume requires a volume asset"
        if node.input_structure in {"image-text", "paired-study"} and not relations.get("paired" if node.input_structure == "image-text" else "paired_study", 0):
            return False, f"{node.input_structure} requires explicit paired evidence"
        if node.operation == "localize" and not (evidence.get("region", 0) or evidence.get("index", 0)):
            return False, "localize requires region or index evidence"
        if node.operation == "quantify" and not (evidence.get("numeric", 0) or context.get("computable_numeric")):
            return False, "quantify requires numeric or computable evidence"
        if node.operation == "compare" and relations.get("comparable", 0) < 2:
            return False, "compare requires two linked comparable evidence items"
        if node.operation == "align" and relations.get("related", 0) < 2:
            return False, "align requires two related evidence objects"
        if node.operation == "describe" and not (evidence.get("text", 0) or relations.get("paired_text", 0) or context.get("structured_reference")):
            return False, "describe requires paired text or structured reference"
        return True, None


def build_task_nodes(axis_analysis: AxisAnalysis, context: dict[str, Any], *, registry=None, exclusions=None) -> dict[str, TaskNode]:
    registry = registry or NodeConstraintRegistry()
    supported = {
        TaskAxis.INPUT_STRUCTURE: [x.value for x in axis_analysis.input_structure if x.supported],
        TaskAxis.OPERATION: [x.value for x in axis_analysis.operation if x.supported],
        TaskAxis.CLINICAL_TARGET: [x.value for x in axis_analysis.clinical_target if x.supported],
    }
    nodes: dict[str, TaskNode] = {}
    for input_structure, operation, clinical_target in product(
        supported[TaskAxis.INPUT_STRUCTURE], supported[TaskAxis.OPERATION], supported[TaskAxis.CLINICAL_TARGET]
    ):
        node = TaskNode(
            node_id=stable_node_id(input_structure, operation, clinical_target),
            input_structure=input_structure, operation=operation, clinical_target=clinical_target,
            capability_category=CAPABILITY_BY_OPERATION[operation], evidence_stats=context.get("evidence", {}),
        )
        feasible, reason = registry.check(node, context)
        if feasible:
            nodes[node.node_id] = node
        else:
            if exclusions is not None:
                exclusions[node.node_id] = {"node": node.model_dump(), "reason": reason}
            nodes[node.node_id] = node.model_copy(update={"exclusion_reason": reason})
    return nodes

