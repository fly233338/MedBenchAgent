"""Stable Task Planning vocabulary and validated task-space contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskAxis(StrEnum):
    INPUT_STRUCTURE = "input_structure"
    OPERATION = "operation"
    CLINICAL_TARGET = "clinical_target"


AXIS_VALUES: dict[TaskAxis, tuple[str, ...]] = {
    TaskAxis.INPUT_STRUCTURE: (
        "single-image", "multi-image", "multi-view", "volume", "image-text", "paired-study",
    ),
    TaskAxis.OPERATION: ("recognize", "localize", "quantify", "compare", "align", "describe"),
    TaskAxis.CLINICAL_TARGET: (
        "acquisition", "anatomy", "finding", "attribute", "measurement", "assessment", "diagnosis", "quality",
    ),
}


# This is the structural operation family. It is deliberately separate from
# the post-planning Blueprint capability dimensions.
OPERATION_FAMILY_BY_OPERATION = {
    "recognize": "Recognition", "localize": "Localization", "quantify": "Quantification",
    "compare": "Comparison", "align": "Integration", "describe": "Generation",
}
# Keep the old constant as a source-compatible alias for downstream callers.
CAPABILITY_BY_OPERATION = OPERATION_FAMILY_BY_OPERATION


class TaskParadigm(StrEnum):
    VQA = "VQA"
    VISUAL_GROUNDING = "Visual Grounding"
    REPORT_GENERATION = "Report Generation"


class OutputSpec(StrEnum):
    LABEL = "label"
    LABELS = "labels"
    NUMBER = "number"
    TEXT = "text"
    BBOX = "bbox"
    INDEX = "index"


class CapabilityCategory(StrEnum):
    """Legacy operation-family enum retained for metric configuration."""

    RECOGNITION = "Recognition"
    LOCALIZATION = "Localization"
    QUANTIFICATION = "Quantification"
    COMPARISON = "Comparison"
    INTEGRATION = "Integration"
    GENERATION = "Generation"


class EvidenceBasis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reference: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class EvidenceSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str | None = None
    evidence_type: Literal["value", "text", "asset", "region"] | None = None
    relation: str | None = None
    role: str = Field(min_length=1)
    unit: str | None = None


class AxisAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str = Field(min_length=1)
    supported: bool
    evidence_basis: list[EvidenceBasis] = Field(min_length=1)
    reason: str = Field(min_length=1)


class AxisAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_structure: list[AxisAssessment]
    operation: list[AxisAssessment]
    clinical_target: list[AxisAssessment]
    dataset_revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self):
        for axis, values in (
            (TaskAxis.INPUT_STRUCTURE, self.input_structure),
            (TaskAxis.OPERATION, self.operation),
            (TaskAxis.CLINICAL_TARGET, self.clinical_target),
        ):
            allowed = set(AXIS_VALUES[axis])
            actual = [item.value for item in values]
            if set(actual) != allowed or len(actual) != len(set(actual)):
                raise ValueError(f"Axis analysis must cover registry exactly for {axis}")
        return self


class TaskNode(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    node_id: str = Field(min_length=1)
    input_structure: str
    operation: str
    clinical_target: str
    operation_family: str | None = None
    # Compatibility field for existing manifests and metric configuration.
    capability_category: str | None = None
    annotation_mapping: list[EvidenceSelector] = Field(default_factory=list)
    evidence_stats: dict[str, Any] = Field(default_factory=dict)
    feasible_case_count: int | None = Field(default=None, ge=0)
    exclusion_reason: str | None = None
    leaf_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "active", "review_pending", "completed", "blocked", "excluded"] = "pending"
    capability_dimension_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_values(self):
        if self.input_structure not in AXIS_VALUES[TaskAxis.INPUT_STRUCTURE]:
            raise ValueError("Unknown input structure")
        if self.operation not in AXIS_VALUES[TaskAxis.OPERATION]:
            raise ValueError("Unknown operation")
        if self.clinical_target not in AXIS_VALUES[TaskAxis.CLINICAL_TARGET]:
            raise ValueError("Unknown clinical target")
        family = OPERATION_FAMILY_BY_OPERATION[self.operation]
        if self.operation_family is None:
            self.operation_family = family
        if self.capability_category is None:
            self.capability_category = family
        if self.operation_family != family or self.capability_category != family:
            raise ValueError("Operation family must follow the operation mapping")
        if self.exclusion_reason and self.status == "pending":
            self.status = "excluded"
        if self.status == "excluded" and not self.exclusion_reason:
            raise ValueError("Excluded Node requires an exclusion reason")
        if len(set(self.leaf_ids)) != len(self.leaf_ids):
            raise ValueError("Task Node leaf IDs must be unique")
        return self


class TaskLeaf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leaf_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    # question is retained as a compatibility alias. New plans use the
    # canonical question_template field and generation reads that field.
    question: str | None = None
    question_template: str | None = None
    input: list[EvidenceSelector] = Field(min_length=1)
    target: str = Field(min_length=1)
    reference_rule: str | None = None
    answer: str | None = None
    resolver: dict[str, Any]
    output_spec: OutputSpec
    task_paradigm: TaskParadigm
    capability_category: str = Field(min_length=1)
    capability_dimension_ids: list[str] = Field(default_factory=list)
    evidence: list[EvidenceBasis] = Field(default_factory=list)
    eligible_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_contract(self):
        template = self.question_template or self.question
        if not template or not template.strip():
            raise ValueError("Task Leaf requires question_template")
        self.question_template = template.strip()
        self.question = self.question or self.question_template
        self.reference_rule = self.reference_rule or self.answer
        allowed = {
            TaskParadigm.VQA: {OutputSpec.LABEL, OutputSpec.LABELS, OutputSpec.NUMBER, OutputSpec.TEXT},
            TaskParadigm.VISUAL_GROUNDING: {OutputSpec.BBOX, OutputSpec.INDEX},
            TaskParadigm.REPORT_GENERATION: {OutputSpec.TEXT},
        }
        if self.output_spec not in allowed[self.task_paradigm]:
            raise ValueError("OutputSpec is incompatible with TaskParadigm")
        return self


class NodeReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    node_id: str = Field(min_length=1)
    checked_leaf_ids: list[str] = Field(min_length=1)
    status: Literal["approved", "rejected"]
    issues: list[str] = Field(default_factory=list)


class TaskSpacePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "task-space-plan"
    dataset: dict[str, Any]
    axis_analysis: AxisAnalysis | None = None
    nodes: dict[str, TaskNode] = Field(default_factory=dict)
    leaves: dict[str, TaskLeaf] = Field(default_factory=dict)
    dataset_facts: dict[str, Any] = Field(default_factory=dict)
    blueprint: dict[str, Any] = Field(default_factory=dict)
    axis_assessments: dict[str, list[AxisAssessment]] = Field(default_factory=dict)
    phase: Literal["facts", "axes", "nodes", "leaves", "capabilities", "complete", "blocked"] = "facts"
    next_axis: TaskAxis | None = TaskAxis.INPUT_STRUCTURE
    current_node_id: str | None = None
    pending_node_ids: list[str] = Field(default_factory=list)
    completed_node_ids: list[str] = Field(default_factory=list)
    reviews: dict[str, NodeReview] = Field(default_factory=dict)
    capability_assignments: dict[str, list[str]] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)

    @field_validator("next_axis", mode="before")
    @classmethod
    def parse_next_axis(cls, value):
        return None if value is None else TaskAxis(value)


def stable_node_id(input_structure: str, operation: str, clinical_target: str) -> str:
    payload = json.dumps([input_structure, operation, clinical_target], separators=(",", ":"))
    return "node-" + hashlib.sha256(payload.encode()).hexdigest()[:16]
