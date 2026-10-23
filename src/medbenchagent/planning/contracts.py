"""Semantic planning products, without a Resolver or execution state."""

from typing import Literal

from pydantic import Field, model_validator

from medbenchagent.data.dataset import DatasetRef
from medbenchagent.data.models import CanonicalModel, EvidenceRef
from medbenchagent.data.task_data import ProductBinding


TaskFamily = Literal[
    "closed_vqa", "open_vqa", "report_generation", "visual_grounding", "detection"
]


class ProfileEvidenceRef(CanonicalModel):
    observation_id: str = Field(min_length=1)


PlanningEvidenceRef = EvidenceRef | ProfileEvidenceRef


class LeafAssembly(CanonicalModel):
    backend_id: Literal["vqa", "report_generation", "visual_grounding", "detection"]
    input_fields: list[str] = Field(min_length=1)
    question_templates: list[str] = Field(min_length=1)
    question_fields: dict[str, str] = Field(default_factory=dict)
    reference_field: str = Field(min_length=1)
    output_spec: Literal["label", "labels", "number", "text", "bbox", "index", "detections"]
    options: list[str] | None = None
    label_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bindings(self):
        from string import Formatter
        if len(self.input_fields) != len(set(self.input_fields)):
            raise ValueError("Leaf input fields must be unique and ordered")
        if self.options is not None and (len(self.options) < 2 or len(self.options) != len(set(self.options))):
            raise ValueError("Options must have at least two unique values")
        if self.options is not None and any(not value.strip() for value in self.options):
            raise ValueError("Options cannot be blank")
        if any(not name.isidentifier() or name in {"options", "task_objective"} for name in self.question_fields):
            raise ValueError("Question variables must be identifiers and cannot replace reserved placeholders")
        if self.label_map and (self.options is None or set(self.label_map.values()) - set(self.options)):
            raise ValueError("Label mapping must target declared options")
        allowed = {"options", "task_objective", *self.question_fields}
        for text in self.question_templates:
            if not text.strip():
                raise ValueError("Empty question template")
            for _, name, formatting, conversion in Formatter().parse(text):
                if name is not None and (name not in allowed or formatting or conversion):
                    raise ValueError("Templates only support declared named placeholders")
        return self


class RequiredField(CanonicalModel):
    name: str = Field(min_length=1)
    type: Literal["asset", "text", "value"]
    input_visibility: Literal["model_visible", "reference_only", "unknown"]
    purpose: str = Field(min_length=1)


class DataRequirement(CanonicalModel):
    requirement_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    dataset: DatasetRef
    case_ids: list[str] = Field(default_factory=list)
    purpose: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    evidence_refs: list[PlanningEvidenceRef] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)
    affected_task_ids: list[str] = Field(default_factory=list)
    affected_scenario_ids: list[str] = Field(default_factory=list)
    backend_id: str | None = None
    source_role: Literal["task_planner", "generation"] = "task_planner"
    required_fields: list[RequiredField] = Field(default_factory=list)
    case_granularity: str = Field(default="source Case", min_length=1)
    input_relationships: list[str] = Field(default_factory=list)
    minimum_cases: int = Field(default=1, ge=1)


class SemanticTaskSpec(CanonicalModel):
    task_id: str = Field(min_length=1)
    dataset: DatasetRef
    scenario_id: str = Field(min_length=1)
    task_family: TaskFamily
    objective: str = Field(min_length=1)
    input_requirements: list[str] = Field(min_length=1)
    reference_requirements: list[str] = Field(min_length=1)
    output_paradigm: str = Field(min_length=1)
    evidence_refs: list[PlanningEvidenceRef] = Field(default_factory=list)
    feasibility: Literal["proposed", "needs_data", "feasible", "infeasible"] = "proposed"
    requirement_ids: list[str] = Field(default_factory=list)
    revision: int = Field(default=1, ge=1)
    change_reason: str = Field(default="Initial task", min_length=1)
    input_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    reference_evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    task_data: ProductBinding | None = None
    assembly: LeafAssembly | None = None
    capability_category: Literal["Recognition", "Localization", "Quantification", "Comparison", "Integration", "Generation"] | None = None


Axis = Literal["input_structure", "operation", "clinical_target"]


class AxisAssessment(CanonicalModel):
    value: str = Field(min_length=1)
    supported: bool
    reason: str = Field(min_length=1)
    evidence_refs: list[PlanningEvidenceRef] = Field(min_length=1)


class SemanticScenario(CanonicalModel):
    scenario_id: str = Field(min_length=1)
    input_structure: str
    operation: str
    clinical_target: str
    decision: Literal["pending", "keep", "exclude"] = "pending"
    reason: str = ""
    evidence_refs: list[PlanningEvidenceRef] = Field(default_factory=list)
    backend_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)


class TaskExclusion(CanonicalModel):
    reason: str = Field(min_length=1)
    evidence_refs: list[PlanningEvidenceRef] = Field(min_length=1)


class RequirementAcceptance(CanonicalModel):
    artifact_id: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1)
    evidence_refs: list[PlanningEvidenceRef] = Field(min_length=1)


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
    task_data_acceptances: dict[str, RequirementAcceptance] = Field(default_factory=dict)
    exclusions: dict[str, TaskExclusion] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identities(self):
        for items, key in ((self.tasks, "task_id"), (self.requirements, "requirement_id"),
                           (self.scenarios, "scenario_id")):
            if any(identifier != getattr(item, key) for identifier, item in items.items()):
                raise ValueError("Plan dictionary keys must equal object identities")
        if set(self.exclusions) - self.tasks.keys() or set(self.acceptances) - self.requirements.keys():
            raise ValueError("Unknown task exclusion or requirement acceptance")
        if any(key != value.artifact_id for key, value in self.task_data_acceptances.items()):
            raise ValueError("TaskData acceptance keys must match their artifact IDs")
        return self
