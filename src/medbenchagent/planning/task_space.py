"""Stable Task Discovery vocabulary and validated task-space contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskAxis(StrEnum):
    INPUT_STRUCTURE = "input_structure"
    OPERATION = "operation"
    CLINICAL_TARGET = "clinical_target"


AXIS_VALUES: dict[TaskAxis, tuple[str, ...]] = {
    TaskAxis.INPUT_STRUCTURE: ("single-image", "multi-image", "multi-view", "volume", "image-text", "paired-study"),
    TaskAxis.OPERATION: ("recognize", "localize", "quantify", "compare", "align", "describe"),
    TaskAxis.CLINICAL_TARGET: ("acquisition", "anatomy", "finding", "attribute", "measurement", "assessment", "diagnosis", "quality"),
}

CAPABILITY_BY_OPERATION = {
    "recognize": "Recognition", "localize": "Localization", "quantify": "Quantification",
    "compare": "Comparison", "align": "Integration", "describe": "Generation",
}


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
        for axis, values in ((TaskAxis.INPUT_STRUCTURE, self.input_structure),
                             (TaskAxis.OPERATION, self.operation),
                             (TaskAxis.CLINICAL_TARGET, self.clinical_target)):
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
    capability_category: str
    evidence_stats: dict[str, Any] = Field(default_factory=dict)
    exclusion_reason: str | None = None
    leaf_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_values(self):
        if self.input_structure not in AXIS_VALUES[TaskAxis.INPUT_STRUCTURE]:
            raise ValueError("Unknown input structure")
        if self.operation not in AXIS_VALUES[TaskAxis.OPERATION]:
            raise ValueError("Unknown operation")
        if self.clinical_target not in AXIS_VALUES[TaskAxis.CLINICAL_TARGET]:
            raise ValueError("Unknown clinical target")
        if self.capability_category != CAPABILITY_BY_OPERATION[self.operation]:
            raise ValueError("Capability category must follow the operation mapping")
        return self


class EvidenceSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str | None = None
    evidence_type: str | None = None
    relation: str | None = None
    role: str = Field(min_length=1)


class TaskLeaf(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leaf_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    input: list[EvidenceSelector] = Field(min_length=1)
    target: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    resolver: dict[str, Any]
    output_spec: OutputSpec
    task_paradigm: TaskParadigm
    capability_category: str = Field(min_length=1)
    evidence: list[EvidenceBasis] = Field(default_factory=list)
    eligible_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_paradigm_output(self):
        allowed = {
            TaskParadigm.VQA: {OutputSpec.LABEL, OutputSpec.LABELS, OutputSpec.NUMBER, OutputSpec.TEXT},
            TaskParadigm.VISUAL_GROUNDING: {OutputSpec.BBOX, OutputSpec.INDEX},
            TaskParadigm.REPORT_GENERATION: {OutputSpec.TEXT},
        }
        if self.output_spec not in allowed[self.task_paradigm]:
            raise ValueError("OutputSpec is incompatible with TaskParadigm")
        return self


class TaskSpacePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "task-space-plan.v1"
    dataset: dict[str, Any]
    axis_analysis: AxisAnalysis
    nodes: dict[str, TaskNode]
    leaves: dict[str, TaskLeaf]
    scope: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)


def stable_node_id(input_structure: str, operation: str, clinical_target: str) -> str:
    payload = json.dumps([input_structure, operation, clinical_target], separators=(",", ":"))
    return "node-" + hashlib.sha256(payload.encode()).hexdigest()[:16]

