"""Canonical contracts for the three preparation coding stages.

The worker boundary is deliberately small. A worker produces a candidate
submission; the host owns stage validation, handoff acceptance and
publication.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from medbenchagent.data.models import PresentationSpec


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Stage(StrEnum):
    UNDERSTAND = "understand"
    PREPARE_TEXT = "prepare_text"
    PREPARE_IMAGE = "prepare_image"


CONTROL_MAPPING_REVISION_REQUIRED = "mapping_revision_required"
class ArtifactRef(Contract):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str = Field(min_length=1)


class PopulationPolicy(Contract):
    rule_id: str = Field(min_length=1)
    unit_type: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    identity_fields: list[str] = Field(min_length=1)
    inclusion: str = Field(min_length=1)
    exclusion: str = ""
    evidence_refs: list[str] = Field(min_length=1)


class CaseIdentityRule(Contract):
    rule_id: str = Field(min_length=1)
    target_path: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)
    collision_policy: Literal["error", "merge", "split"] = "error"
    evidence_refs: list[str] = Field(min_length=1)


class GroupingRule(Contract):
    rule_id: str = Field(min_length=1)
    case_unit: str = Field(min_length=1)
    key_fields: list[str] = Field(min_length=1)
    conflict_policy: Literal["error", "separate", "prefer_source", "record_all"] = "error"
    evidence_refs: list[str] = Field(min_length=1)
    decision_id: str | None = None


class JoinRule(Contract):
    rule_id: str = Field(min_length=1)
    left_scope: str = Field(min_length=1)
    right_scope: str = Field(min_length=1)
    keys: list[str] = Field(min_length=1)
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    null_key_policy: Literal["reject", "unmatched", "match_null"] = "unmatched"
    duplicate_key_policy: Literal["reject", "all", "first"] = "reject"
    unmatched_policy: Literal["reject", "quarantine", "exclude"] = "reject"
    evidence_refs: list[str] = Field(min_length=1)


class FieldRule(Contract):
    rule_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    disposition: Literal[
        "medical_evidence", "grouping_metadata", "provenance_metadata",
        "asset_metadata", "exclude",
    ]
    target_path: str = ""
    source_type: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    raw_value_path: str = ""
    transform: str = "identity"
    missingness: str = Field(min_length=1)
    unit: str | None = None
    reason: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    decision_id: str | None = None


class StageBudget(Contract):
    max_iterations: int = Field(default=60, ge=1)
    timeout_seconds: int = Field(default=1800, ge=1)
    initial_message_tokens: int = Field(default=12_000, ge=256)
    model_context_tokens: int | None = Field(default=None, ge=256)
    output_reserve_tokens: int = Field(default=4_096, ge=0)
    conversation_reserve_tokens: int = Field(default=8_192, ge=0)
    safety_margin_tokens: int = Field(default=1_024, ge=0)


class Diagnostic(Contract):
    code: str = Field(min_length=1)
    stage: Stage
    message: str = Field(min_length=1)
    issue_ids: list[str] = Field(default_factory=list)
    path: str | None = None
    record_ref: str | None = None
    field: str | None = None
    category: Literal["implementation", "mapping", "schema", "population", "environment", "contract", "unknown"] = "unknown"
    blocking: bool = False
    rule_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    observed: str = ""
    expected: str = ""
    affected_artifacts: list[str] = Field(default_factory=list)
    requested_change: str = ""


class SemanticQuestion(Contract):
    issue_id: str = Field(min_length=1)
    stage: Stage
    source_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    question: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    impact: str = Field(min_length=1)
    blocking: bool = False
    resolved: bool = False
    decision_id: str | None = None


class ImageRule(Contract):
    rule_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    pattern: str = "*"
    path_field: str | None = None
    source_format: str = Field(min_length=1)
    action: Literal["direct", "convert", "reference_only"]
    policy_id: str | None = None
    notes: str = ""


class RecordSource(Contract):
    source_id: str
    path: str
    format: Literal["csv", "tsv", "xlsx", "xls", "jsonl", "json"]
    sheet: str | int | None = None
    records_field: str | None = None


class PreparationMapping(Contract):
    population_policy: PopulationPolicy
    case_identity: CaseIdentityRule
    grouping: GroupingRule
    joins: list[JoinRule] = Field(default_factory=list)
    fields: list[FieldRule] = Field(min_length=1)
    image_rules: list[ImageRule] = Field(min_length=1)
    image_overrides: dict[str, ImageRule] = Field(default_factory=dict)
    record_sources: list[RecordSource] = Field(min_length=1)
    field_semantics: dict[str, Any] = Field(default_factory=dict)
    resolved_decision_refs: list[str] = Field(default_factory=list)
    limitations: list[Any] = Field(default_factory=list)


class UserDecision(Contract):
    decision_id: str = Field(min_length=1)
    issue_id: str = Field(min_length=1)
    dataset_id: str = ""
    source_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    answer: str = Field(min_length=1)
    exchanges: list[dict[str, str]] = Field(default_factory=list)
    source_snapshot_digest: str = ""
    input_digest: str = ""
    structured_value: Any = None
    evidence_refs: list[str] = Field(default_factory=list)
    digest: str = ""

    @model_validator(mode="after")
    def bind_digest(self):
        value = self.model_dump(exclude={"digest"}, mode="json")
        expected = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if self.digest and self.digest != expected:
            raise ValueError("User decision digest does not match its contents")
        self.digest = expected
        return self


class AcceptedHandoff(Contract):
    dataset_id: str
    source_snapshot_digest: str
    mapping: ArtifactRef
    image_conversion_plan: ArtifactRef
    digest: str


class ValidationResult(Contract):
    status: Literal["accepted", "rejected", "needs_input", "blocked"]
    stage: Stage
    issues: list[Diagnostic] = Field(default_factory=list)
    questions: list[SemanticQuestion] = Field(default_factory=list)
    candidate_digest: str = ""


class Submission(Contract):
    """Worker's explicit candidate handoff; it is not host acceptance."""

    requested_outcome: Literal["submitted", "needs_input", "blocked"]
    summary: str = ""
    outputs: list[str] = Field(default_factory=list)
    question_refs: list[str] = Field(default_factory=list)
    diagnostic_codes: list[str] = Field(default_factory=list)
    questions: list[SemanticQuestion] = Field(default_factory=list)


class StageRequest(Contract):
    stage: Stage
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    attempt_id: str = ""
    conversation_id: str = Field(min_length=1)
    execution_key: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    prompt_path: str = Field(min_length=1)
    context_path: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    source_roots: dict[str, str] = Field(default_factory=dict)
    source_manifests: dict[str, str] = Field(default_factory=dict)
    input_digest: str = Field(min_length=1)
    source_snapshot_digest: str = ""
    prompt_token_count: int | None = Field(default=None, ge=0)
    required_outputs: list[str] = Field(default_factory=list)
    optional_outputs: list[str] = Field(default_factory=list)
    budget: StageBudget = Field(default_factory=StageBudget)
    construction_events_path: str = ""
    model: str = ""
    base_url: str = ""
    reasoning_effort: str = "medium"
    llm_timeout_seconds: float | None = Field(default=None, ge=0.000001)
    llm_max_retries: int | None = Field(default=None, ge=0)
    agent: str = "DatasetProcessingAgent"
    resume_existing: bool = False
    validation_feedback: dict[str, Any] = Field(default_factory=dict)
    runtime_backend: str = ""
    previous_attempt_root: str | None = None
    worker_runtime: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_paths(self):
        if not self.required_outputs:
            raise ValueError("A stage request must declare required outputs")
        return self


class StageResult(Contract):
    stage: Stage
    run_id: str = ""
    dataset_id: str = ""
    attempt: int = Field(ge=1)
    attempt_id: str = ""
    conversation_id: str = Field(min_length=1)
    execution_key: str = Field(min_length=1)
    status: Literal["submitted", "needs_input", "blocked", "failed", "cancelled"]
    control: Literal["mapping_revision_required"] | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    question_refs: list[str] = Field(default_factory=list)
    questions: list[SemanticQuestion] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    diagnostic: str = ""
    termination_reason: str = ""
    progress: dict[str, Any] = Field(default_factory=dict)
    submission_path: str | None = None
    input_digest: str = ""
    prompt_digest: str = ""
    context_digest: str = ""
    token_count: int | None = Field(default=None, ge=0)
    usage: dict[str, Any] = Field(default_factory=dict)
    execution_refs: list[str] = Field(default_factory=list)


STAGE_OUTPUTS = {
    Stage.UNDERSTAND: ("mapping.json", "image_conversion_plan.json"),
    Stage.PREPARE_TEXT: ("cases.jsonl",),
    Stage.PREPARE_IMAGE: ("image_manifest.jsonl",),
}




class ImageConversionPlan(Contract):
    policies: dict[str, PresentationSpec] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def digest_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
