"""Semantic planning products, without a Resolver or execution state."""

from typing import Literal

from pydantic import Field

from medbenchagent.data.dataset import DatasetRef
from medbenchagent.data.models import CanonicalModel, EvidenceRef


TaskFamily = Literal[
    "closed_vqa", "open_vqa", "report_generation", "visual_grounding", "detection"
]


class DataRequirement(CanonicalModel):
    requirement_id: str = Field(min_length=1)
    dataset: DatasetRef
    case_ids: list[str] = Field(default_factory=list)
    purpose: str = Field(min_length=1)
    representation: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)


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
