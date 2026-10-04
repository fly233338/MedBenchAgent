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

    @field_validator("axis", mode="before")
    @classmethod
    def parse_axis(cls, value):
        return TaskAxis(value)

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


class ValidatedClinicalTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    reference_kind: Literal["direct", "derived"]
    reference_fields: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference_fields(self):
        if len(set(self.reference_fields)) != len(self.reference_fields):
            raise ValueError("Validated clinical target reference_fields must be unique")
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
    screening_decision: Literal["keep", "exclude"] | None = None
    screening_reason: str | None = None
    screening_supporting_fields: list[str] = Field(default_factory=list)
    validated_clinical_targets: list[ValidatedClinicalTarget] = Field(default_factory=list)
    exclusion_reason: str | None = None
    exclusion_stage: Literal["structural", "screening", "leaf_review"] | None = None
    exclusion_evidence: list[EvidenceBasis] = Field(default_factory=list)
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
        if self.screening_decision == "keep" and self.exclusion_reason and self.exclusion_stage != "leaf_review":
            raise ValueError("A kept Scenario cannot have an exclusion reason")
        if self.screening_decision == "keep" and not self.validated_clinical_targets:
            raise ValueError("Keeping a Scenario requires validated clinical targets")
        if self.exclusion_reason and self.exclusion_stage is None:
            self.exclusion_stage = "screening"
        if self.exclusion_stage and not self.exclusion_reason:
            raise ValueError("Scenario exclusion stage requires an exclusion reason")
        if len(set(self.screening_supporting_fields)) != len(self.screening_supporting_fields):
            raise ValueError("Scenario screening supporting fields must be unique")
        target_ids = [item.target_id for item in self.validated_clinical_targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("Scenario validated clinical target IDs must be unique")
        if any(item.target_type != self.clinical_target for item in self.validated_clinical_targets):
            raise ValueError("Validated clinical target type must match Scenario clinical_target")
        if len(set(self.leaf_task_ids)) != len(self.leaf_task_ids):
            raise ValueError("Scenario Leaf Task IDs must be unique")
        return self

class LeafTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    leaf_task_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    clinical_target_id: str = Field(min_length=1)
    question_templates: list[str] = Field(min_length=1)
    options: list[str] | None = None
    input: list[EvidenceSelector] = Field(min_length=1)
    target: str = Field(min_length=1)
    reference_rule: str = Field(min_length=1)
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

    @field_validator("output_spec", mode="before")
    @classmethod
    def parse_output_spec(cls, value):
        return OutputSpec(value)

    @field_validator("task_paradigm", mode="before")
    @classmethod
    def parse_task_paradigm(cls, value):
        return TaskParadigm(value)

    @model_validator(mode="after")
    def validate_contract(self):
        if any(not template.strip() for template in self.question_templates):
            raise ValueError("LeafTask question_templates must be non-empty")
        if any("{{" in template or "}}" in template for template in self.question_templates):
            raise ValueError("LeafTask question_templates must be complete; unresolved placeholders are not supported")
        if len(set(self.question_templates)) != len(self.question_templates):
            raise ValueError("LeafTask question_templates must be unique")
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
    options: list[str] | None = None
    answer: Any
    input_bindings: list[dict[str, Any]] = Field(default_factory=list)
    input_assets: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)


class TaskPlanState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "task-plan"
    dataset: dict[str, Any]
    dataset_facts: dict[str, Any] = Field(default_factory=dict)
    blueprint: dict[str, Any] = Field(default_factory=dict)
    access_path: str | None = None
    evidence_manifest_path: str | None = None
    axis_decisions: dict[str, list[AxisValueDecision]] = Field(default_factory=dict)
    scenarios: dict[str, TaskScenario] = Field(default_factory=dict)
    leaf_tasks: dict[str, LeafTask] = Field(default_factory=dict)
    phase: Literal["access", "overview", "axes", "scenarios", "screening", "scenario_review", "leaves", "complete", "blocked"] = "access"
    next_axis: TaskAxis | None = TaskAxis.INPUT_STRUCTURE
    current_scenario_id: str | None = None
    pending_scenario_ids: list[str] = Field(default_factory=list)
    screening_pending_scenario_ids: list[str] = Field(default_factory=list)
    scenario_review_pending_ids: list[str] = Field(default_factory=list)
    scenario_review_completed_ids: list[str] = Field(default_factory=list)
    scenario_review_excluded_ids: list[str] = Field(default_factory=list)
    completed_scenario_ids: list[str] = Field(default_factory=list)
    reviews: dict[str, ScenarioReview] = Field(default_factory=dict)
    extension_selection: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)
    review: dict[str, Any] = Field(default_factory=dict)

    @field_validator("next_axis", mode="before")
    @classmethod
    def parse_next_axis(cls, value):
        return None if value is None else TaskAxis(value)

def stable_scenario_id(input_structure: str, operation: str, clinical_target: str) -> str:
    payload = json.dumps([input_structure, operation, clinical_target], separators=(",", ":"))
    return "scenario-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

