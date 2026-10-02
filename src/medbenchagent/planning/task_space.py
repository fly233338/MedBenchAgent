"""Validated contracts for the six-level task-planning model.

The model may select Axis Values and propose Leaf Tasks. Code owns Scenario
construction, capability tagging, resolver validation, capacity and item
instantiation. Compatibility aliases keep in-process callers usable while
persisted artifacts move to the v3 schema.
"""

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


class CapabilityCategory(StrEnum):
    RECOGNITION = "Recognition"
    LOCALIZATION = "Localization"
    QUANTIFICATION = "Quantification"
    COMPARISON = "Comparison"
    INTEGRATION = "Integration"
    GENERATION = "Generation"


CAPABILITY_BY_OPERATION: dict[str, CapabilityCategory] = {
    "recognize": CapabilityCategory.RECOGNITION,
    "localize": CapabilityCategory.LOCALIZATION,
    "quantify": CapabilityCategory.QUANTIFICATION,
    "compare": CapabilityCategory.COMPARISON,
    "align": CapabilityCategory.INTEGRATION,
    "describe": CapabilityCategory.GENERATION,
}
OPERATION_FAMILY_BY_OPERATION = CAPABILITY_BY_OPERATION


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


class AxisValueDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: str = Field(min_length=1)
    supported: bool
    evidence_basis: list[EvidenceBasis] = Field(min_length=1)
    reason: str = Field(min_length=1)


class AxisDecisionSet(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    axis: TaskAxis
    values: list[AxisValueDecision] = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self):
        allowed = set(AXIS_VALUES[self.axis])
        actual = [item.value for item in self.values]
        if set(actual) != allowed or len(actual) != len(set(actual)):
            raise ValueError(f"Axis decision must cover registry exactly for {self.axis}")
        return self


class AxisAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_structure: list[AxisValueDecision]
    operation: list[AxisValueDecision]
    clinical_target: list[AxisValueDecision]
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


class TaskScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scenario_id: str = Field(min_length=1)
    input_structure: str
    operation: str
    clinical_target: str
    capability_category: CapabilityCategory
    annotation_mapping: list[EvidenceSelector] = Field(default_factory=list)
    evidence_stats: dict[str, Any] = Field(default_factory=dict)
    feasible_case_count: int | None = Field(default=None, ge=0)
    exclusion_reason: str | None = None
    leaf_task_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "active", "review_pending", "completed", "blocked", "excluded"] = "pending"

    @field_validator("capability_category", mode="before")
    @classmethod
    def parse_capability_category(cls, value):
        return CapabilityCategory(value)

    @model_validator(mode="after")
    def validate_values(self):
        if self.input_structure not in AXIS_VALUES[TaskAxis.INPUT_STRUCTURE]:
            raise ValueError("Unknown input structure")
        if self.operation not in AXIS_VALUES[TaskAxis.OPERATION]:
            raise ValueError("Unknown operation")
        if self.clinical_target not in AXIS_VALUES[TaskAxis.CLINICAL_TARGET]:
            raise ValueError("Unknown clinical target")
        expected = CAPABILITY_BY_OPERATION[self.operation]
        if self.capability_category != expected:
            raise ValueError("Capability Category must follow the Operation mapping")
        if self.exclusion_reason and self.status == "pending":
            self.status = "excluded"
        if self.status == "excluded" and not self.exclusion_reason:
            raise ValueError("Excluded Scenario requires an exclusion reason")
        if len(set(self.leaf_task_ids)) != len(self.leaf_task_ids):
            raise ValueError("Scenario Leaf Task IDs must be unique")
        return self

    @property
    def node_id(self) -> str:
        return self.scenario_id

    @property
    def leaf_ids(self) -> list[str]:
        return self.leaf_task_ids


class LeafTask(BaseModel):
    model_config = ConfigDict(extra="allow")
    leaf_task_id: str | None = None
    name: str = Field(min_length=1)
    scenario_id: str | None = None
    question_template: str | None = None
    input: list[EvidenceSelector] = Field(min_length=1)
    target: str = Field(min_length=1)
    reference_rule: str | None = None
    resolver: dict[str, Any]
    output_spec: OutputSpec
    task_paradigm: TaskParadigm
    capability_category: CapabilityCategory
    evidence: list[EvidenceBasis] = Field(default_factory=list)
    eligible_count: int | None = Field(default=None, ge=0)

    @field_validator("capability_category", mode="before")
    @classmethod
    def parse_capability_category(cls, value):
        return CapabilityCategory(value)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value):
        value = dict(value)
        value.setdefault("leaf_task_id", value.get("leaf_id"))
        value.setdefault("scenario_id", value.get("node_id"))
        value.setdefault("question_template", value.get("question"))
        value.setdefault("reference_rule", value.get("answer"))
        return value

    @model_validator(mode="after")
    def validate_contract(self):
        if not self.leaf_task_id or not self.scenario_id or not self.question_template:
            raise ValueError("LeafTask requires leaf_task_id, scenario_id and question_template")
        if not self.reference_rule:
            raise ValueError("LeafTask requires reference_rule")
        allowed = {
            TaskParadigm.VQA: {OutputSpec.LABEL, OutputSpec.LABELS, OutputSpec.NUMBER, OutputSpec.TEXT},
            TaskParadigm.VISUAL_GROUNDING: {OutputSpec.BBOX, OutputSpec.INDEX},
            TaskParadigm.REPORT_GENERATION: {OutputSpec.TEXT},
        }
        if self.output_spec not in allowed[self.task_paradigm]:
            raise ValueError("OutputSpec is incompatible with TaskParadigm")
        return self

    @property
    def leaf_id(self) -> str:
        return self.leaf_task_id

    @property
    def node_id(self) -> str:
        return self.scenario_id

    @property
    def question(self) -> str:
        return self.question_template


class ScenarioReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scenario_id: str = Field(min_length=1)
    checked_leaf_task_ids: list[str] = Field(default_factory=list)
    status: Literal["approved", "rejected"]
    issues: list[str] = Field(default_factory=list)


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    item_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    leaf_task_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    capability_category: CapabilityCategory
    task_paradigm: TaskParadigm
    output_spec: OutputSpec
    question: str = Field(min_length=1)
    answer: Any
    input_assets: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)


class TaskPlanState(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    schema_version: str = "task-plan.v3"
    dataset: dict[str, Any]
    access_path: str | None = None
    evidence_manifest_path: str | None = None
    axis_decisions: dict[str, list[AxisValueDecision]] = Field(default_factory=dict)
    scenarios: dict[str, TaskScenario] = Field(default_factory=dict)
    leaf_tasks: dict[str, LeafTask] = Field(default_factory=dict)
    phase: Literal["access", "overview", "axes", "scenarios", "leaves", "complete", "blocked"] = "access"
    next_axis: TaskAxis | None = TaskAxis.INPUT_STRUCTURE
    current_scenario_id: str | None = None
    pending_scenario_ids: list[str] = Field(default_factory=list)
    completed_scenario_ids: list[str] = Field(default_factory=list)
    reviews: dict[str, ScenarioReview] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)
    review: dict[str, Any] = Field(default_factory=dict)

    @field_validator("next_axis", mode="before")
    @classmethod
    def parse_next_axis(cls, value):
        return None if value is None else TaskAxis(value)

    @property
    def nodes(self):
        return self.scenarios

    @property
    def leaves(self):
        return self.leaf_tasks

    @property
    def pending_node_ids(self):
        return self.pending_scenario_ids

    @property
    def current_node_id(self):
        return self.current_scenario_id


def stable_scenario_id(input_structure: str, operation: str, clinical_target: str) -> str:
    payload = json.dumps([input_structure, operation, clinical_target], separators=(",", ":"))
    return "scenario-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


AxisAssessment = AxisValueDecision
TaskNode = TaskScenario
TaskLeaf = LeafTask
NodeReview = ScenarioReview
TaskSpacePlan = TaskPlanState
stable_node_id = stable_scenario_id
