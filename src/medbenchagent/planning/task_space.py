"""Validated contracts for the six-level task-planning model.

The model may select Axis Values and propose Leaf Tasks. Code owns Scenario
construction, capability tagging, resolver validation, capacity and item
instantiation. Compatibility aliases keep in-process callers usable while
persisted artifacts use the current schema.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskAxis(StrEnum):
    INPUT_STRUCTURE = "input_structure"
    OPERATION = "operation"
    CLINICAL_TARGET = "clinical_target"


def _default_registry():
    from medbenchagent.planning.bir import RegistrySnapshot
    return RegistrySnapshot.from_directory(Path(__file__).resolve().parents[3] / "config" / "bir_template" / "registry")


# Compatibility views for in-process callers.  The source of truth is the
# experiment Registry; these values are derived from the checked-in template.
_DEFAULT_REGISTRY = _default_registry()
AXIS_VALUES: dict[TaskAxis, tuple[str, ...]] = {
    TaskAxis(axis): _DEFAULT_REGISTRY.axis_values(axis) for axis in _DEFAULT_REGISTRY.axis_order
}


class CapabilityCategory(StrEnum):
    RECOGNITION = "Recognition"
    LOCALIZATION = "Localization"
    QUANTIFICATION = "Quantification"
    COMPARISON = "Comparison"
    INTEGRATION = "Integration"
    GENERATION = "Generation"


CAPABILITY_BY_OPERATION: dict[str, CapabilityCategory] = {
    operation: CapabilityCategory(category)
    for operation, category in _DEFAULT_REGISTRY.capability_by_operation.items()
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
        actual = [item.value for item in self.values]
        if len(actual) != len(set(actual)):
            raise ValueError(f"Axis decision contains duplicate values for {self.axis}")
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
            actual = [item.value for item in values]
            if len(actual) != len(set(actual)):
                raise ValueError(f"Axis analysis contains duplicate values for {axis}")
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
        if not all(isinstance(value, str) and value for value in
                   (self.input_structure, self.operation, self.clinical_target)):
            raise ValueError("Scenario Axis values must be non-empty strings")
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
    question_templates: list[str] = Field(min_length=1)
    options: list[str] | None = None
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
        if not value.get("question_templates"):
            legacy_question = value.get("question_template", value.get("question"))
            if legacy_question:
                value["question_templates"] = [legacy_question]
        value.pop("question_template", None)
        value.pop("question", None)
        value.setdefault("reference_rule", value.get("answer"))
        return value

    @model_validator(mode="after")
    def validate_contract(self):
        if not self.leaf_task_id or not self.scenario_id or not self.question_templates:
            raise ValueError("LeafTask requires leaf_task_id, scenario_id and question_templates")
        if any(not template.strip() for template in self.question_templates):
            raise ValueError("LeafTask question_templates must be non-empty")
        if len(set(self.question_templates)) != len(self.question_templates):
            raise ValueError("LeafTask question_templates must be unique")
        if not self.reference_rule:
            raise ValueError("LeafTask requires reference_rule")
        allowed = set(_DEFAULT_REGISTRY.task_contracts.get("paradigm_output_specs", {}).get(
            self.task_paradigm.value, []
        ))
        if allowed and self.output_spec.value not in allowed:
            raise ValueError("OutputSpec is incompatible with TaskParadigm")
        if self.options is not None:
            if any(not isinstance(option, str) or not option.strip() for option in self.options):
                raise ValueError("VQA options must be non-empty strings")
            if len({option.casefold() for option in self.options}) != len(self.options):
                raise ValueError("VQA options must be unique")
        return self

    @property
    def leaf_id(self) -> str:
        return self.leaf_task_id

    @property
    def node_id(self) -> str:
        return self.scenario_id

    @property
    def question(self) -> str:
        return self.question_templates[0]

    @property
    def question_template(self) -> str:
        """Compatibility view for callers that still expect one template."""
        return self.question_templates[0]


class LeafTaskDiscard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    leaf_task_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ScenarioReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scenario_id: str = Field(min_length=1)
    checked_leaf_task_ids: list[str] = Field(default_factory=list)
    discarded_leaf_tasks: list[LeafTaskDiscard] = Field(default_factory=list)
    status: Literal["approved", "rejected"]
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_leaf_ids(self):
        checked = self.checked_leaf_task_ids
        discarded = [item.leaf_task_id for item in self.discarded_leaf_tasks]
        if len(checked) != len(set(checked)):
            raise ValueError("Scenario review checked_leaf_task_ids must be unique")
        if len(discarded) != len(set(discarded)):
            raise ValueError("Scenario review discarded_leaf_tasks must be unique")
        return self


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
    template_index: int = Field(default=0, ge=0)
    template_count: int = Field(default=1, ge=1)
    answer: Any
    input_assets: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)


class TaskPlanState(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    schema_version: str = "task-plan"
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

