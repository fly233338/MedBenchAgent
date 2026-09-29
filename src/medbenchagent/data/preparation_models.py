"""Versioned contracts for the independent preparation workflow."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LabelValue(Contract):
    value: Any = None
    state: Literal["known", "uncertain", "unknown", "not_assessed", "not_applicable", "missing"]
    scope: str
    owner_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class Observation(Contract):
    value: Any = None
    state: Literal["known", "uncertain", "unknown", "not_assessed", "not_applicable", "missing"]
    scope: str
    owner_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class ArtifactRef(Contract):
    kind: str
    path: str
    digest: str
    version: str = ""


class SourceRef(Contract):
    source_id: str
    version: str
    kind: str
    root: str
    manifest: ArtifactRef


class InventoryItem(Contract):
    file_id: str
    source_id: str
    locator: str
    resolved_path: str
    size: int = Field(ge=0)
    mtime: float
    detected_kind: str
    role: Literal["consume", "supporting", "exclude", "unresolved"] = "unresolved"
    group_id: str = ""
    fingerprint_status: str = "pending"


class FileProfile(Contract):
    file_id: str
    source_id: str
    reader: dict[str, Any] = Field(default_factory=dict)
    statistics: dict[str, Any] = Field(default_factory=dict)
    examples: list[Any] = Field(default_factory=list)
    status: Literal["parsed", "excluded", "failed"] = "parsed"


class FileMapping(Contract):
    file_id: str
    action: Literal["consume", "supporting", "exclude", "unresolved"]
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class FieldMapping(Contract):
    mapping_id: str
    source_id: str
    file_group_id: str
    source_field: dict[str, Any]
    source_scope: str
    target_path: str
    target_field_id: str | None = None
    action: Literal["copy", "map_values", "derive", "preserve_only", "exclude", "unresolved"]
    transform: dict[str, Any] = Field(default_factory=dict)
    missingness: dict[str, str] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class RelationshipPlan(Contract):
    relationship_id: str
    left: str
    right: str
    keys: list[dict[str, str]]
    normalization: dict[str, Any] = Field(default_factory=dict)
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    missing_policy: str
    duplicate_policy: str
    output_entity: str
    finding_grouping: dict[str, Any] = Field(default_factory=dict)


class PopulationMember(Contract):
    identity: str
    source_id: str
    raw_image_id: str | None = None
    locator: str
    status: Literal["selected", "excluded", "quarantined"]
    reason: str = ""


class PreparationPlan(Contract):
    schema_version: Literal["preparation-plan.v1"] = "preparation-plan.v1"
    inputs: list[ArtifactRef] = Field(default_factory=list)
    readers: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[RelationshipPlan] = Field(default_factory=list)
    field_mappings: list[FieldMapping] = Field(default_factory=list)
    population_rules: dict[str, Any] = Field(default_factory=dict)
    output_rules: dict[str, Any] = Field(default_factory=dict)
    sample_rules: dict[str, Any] = Field(default_factory=dict)
    validation_rules: dict[str, Any] = Field(default_factory=dict)
    dependency_requirements: list[str] = Field(default_factory=list)


class PreparationQuestion(Contract):
    question_id: str
    stage: str
    source_ids: list[str]
    mapping_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    raw_examples: list[Any] = Field(default_factory=list)
    question: str
    options: list[dict[str, str]] = Field(default_factory=list)
    impact: str
    blocking: bool = True


class PreparationDecision(Contract):
    question_id: str
    option_id: str | None = None
    rule: dict[str, Any]
    answer_text: str
    source_ids: list[str]
    input_digest: str
    revision: int = Field(ge=1)


class StageReceipt(Contract):
    stage: str
    turn_id: str
    input_digest: str
    output_digest: str
    status: str
    started_at: str
    finished_at: str


class ExecutionReceipt(Contract):
    execution_id: str
    stage: str
    argv: list[str]
    returncode: int | None
    output_digest: str
    status: str


class CandidateRef(Contract):
    candidate_id: str
    root: str
    annotation_path: str
    manifest_path: str
    provenance_path: str
    plan_ref: ArtifactRef
    code_ref: ArtifactRef


class ValidatedCandidateRef(CandidateRef):
    validation_ref: ArtifactRef
    population_ref: ArtifactRef
    replay_mode: Literal["sample", "full"] = "sample"


class RegisteredDatasetRef(Contract):
    dataset_id: str
    bundle_root: str
    version: str
    dataset_entry: dict[str, Any]
    annotation_path: str
    dataset_access_manifest: str
    validation_ref: ArtifactRef
