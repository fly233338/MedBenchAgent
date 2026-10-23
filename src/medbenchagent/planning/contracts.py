"""Semantic planning products, without a Resolver or execution state."""

from typing import Literal

from pydantic import Field, model_validator

from medbenchagent.data.dataset import DatasetRef
from medbenchagent.data.models import CanonicalModel, EvidenceRef


TaskFamily = Literal[
    "closed_vqa", "open_vqa", "report_generation", "visual_grounding", "detection"
]


class DataRequirement(CanonicalModel):
    requirement_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    dataset: DatasetRef
    case_ids: list[str] = Field(default_factory=list)
    purpose: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)
    affected_task_ids: list[str] = Field(default_factory=list)
    source_role: Literal["task_planner", "generation"] = "task_planner"


class SemanticTaskSpec(CanonicalModel):
    task_id: str = Field(min_length=1)
    dataset: DatasetRef
    scenario_id: str = Field(min_length=1)
    task_family: TaskFamily
    objective: str = Field(min_length=1)
    input_requirements: list[str] = Field(min_length=1)
    reference_requirements: list[str] = Field(min_length=1)
    output_paradigm: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    feasibility: Literal["proposed", "needs_data", "feasible", "infeasible"] = "proposed"
    requirement_ids: list[str] = Field(default_factory=list)
    revision: int = Field(default=1, ge=1)
    change_reason: str = Field(default="Initial task", min_length=1)
    input_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    reference_evidence_refs: list[EvidenceRef] = Field(default_factory=list)


Axis = Literal["input_structure", "operation", "clinical_target"]


class AxisAssessment(CanonicalModel):
    value: str = Field(min_length=1)
    supported: bool
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)


class SemanticScenario(CanonicalModel):
    scenario_id: str = Field(min_length=1)
    input_structure: str
    operation: str
    clinical_target: str
    decision: Literal["pending", "keep", "exclude"] = "pending"
    reason: str = ""
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class TaskExclusion(CanonicalModel):
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)


class RequirementAcceptance(CanonicalModel):
    artifact_id: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(min_length=1)


class TaskPlan(CanonicalModel):
    """Business document. No agent messages, phase lock or execution configuration."""

    plan_id: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    dataset: DatasetRef
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    axis_assessments: dict[Axis, list[AxisAssessment]] = Field(default_factory=dict)
    scenarios: dict[str, SemanticScenario] = Field(default_factory=dict)
    tasks: dict[str, SemanticTaskSpec] = Field(default_factory=dict)
    requirements: dict[str, DataRequirement] = Field(default_factory=dict)
    acceptances: dict[str, RequirementAcceptance] = Field(default_factory=dict)
    exclusions: dict[str, TaskExclusion] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identities(self):
        for items, key in ((self.tasks, "task_id"), (self.requirements, "requirement_id"),
                           (self.scenarios, "scenario_id")):
            if any(identifier != getattr(item, key) for identifier, item in items.items()):
                raise ValueError("Plan dictionary keys must equal object identities")
        if set(self.exclusions) - self.tasks.keys() or set(self.acceptances) - self.requirements.keys():
            raise ValueError("Unknown task exclusion or requirement acceptance")
        return self
