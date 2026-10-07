"""Canonical contracts for the two preparation coding stages.

The worker boundary is deliberately small. A worker produces a candidate
submission; the host owns the two validation gates, handoff acceptance and
publication.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Stage(StrEnum):
    UNDERSTAND = "understand"
    PREPARE_TEXT = "prepare_text"
    PREPARE_IMAGE = "prepare_image"


CONTROL_MAPPING_REVISION_REQUIRED = "mapping_revision_required"
SEMANTIC_REVISION_CODES = frozenset({
    "semantic_conflict",
    "population_definition_conflict",
    "grouping_semantics_conflict",
    "field_semantics_conflict",
})


class ArtifactRef(Contract):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: Literal["candidate", "accepted", "invalid", "diagnostic"] = "candidate"
    depends_on: list[str] = Field(default_factory=list)
    purpose: str = ""


class EvidenceRef(Contract):
    ref: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    observation_digest: str = Field(min_length=1)
    excerpt: str = ""


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
    measured: dict[str, int] = Field(default_factory=dict)
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


class AssetRule(Contract):
    rule_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    order: int = Field(ge=0)
    coordinate_system: str = Field(min_length=1)
    transform: str = "identity"
    source_asset_field: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class PopulationRecord(Contract):
    source_id: str = Field(min_length=1)
    source_unit_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    unit_type: str = Field(default="record", min_length=1)
    label_status: Literal["labeled", "unlabeled", "unknown"] = "unknown"
    relation_refs: list[str] = Field(default_factory=list)


class PopulationOutcome(Contract):
    source_id: str = Field(min_length=1)
    source_unit_id: str = Field(min_length=1)
    status: Literal["retained", "quarantined", "excluded", "duplicate"]
    case_ids: list[str] = Field(default_factory=list)
    record_refs: list[str] = Field(default_factory=list)
    locators: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    duplicate_of: str | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "retained" and not self.case_ids:
            raise ValueError("retained population outcomes require case_ids")
        if self.status == "excluded" and not self.rule_ids:
            raise ValueError("excluded population outcomes require rule_ids")
        if self.status == "duplicate" and not self.duplicate_of:
            raise ValueError("duplicate population outcomes require duplicate_of")
        return self


class InputBinding(Contract):
    path: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    role: str = Field(min_length=1)
    read_policy: Literal["before_work", "stream_during_execution", "on_failure", "optional"] = "before_work"
    required: bool = True
    reason: str = ""
    size_bytes: int = Field(default=0, ge=0)
    format: str = "file"
    schema_ref: str | None = None
    producer: str = "host"
    depends_on: list[str] = Field(default_factory=list)
    host_path: str | None = None
    worker_path: str | None = None


class DigestEntry(Contract):
    digest: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)


class SourceBinding(Contract):
    source_id: str = Field(min_length=1)
    root: str = Field(min_length=1)
    manifest: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    read_only: bool = True


class StageBudget(Contract):
    max_iterations: int = Field(default=40, ge=1)
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


class QuestionSet(Contract):
    questions: list[SemanticQuestion]

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.issue_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic question issue_id values must be unique")
        return self


class ArtifactDependency(Contract):
    path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    digest: str = ""


class OutputFile(Contract):
    path: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    producer: str = "worker"
    purpose: str = ""
    depends_on: list[str] = Field(default_factory=list)


class OutputManifest(Contract):
    files: list[OutputFile] = Field(default_factory=list)
    dependencies: list[ArtifactDependency] = Field(default_factory=list)
    mapping_digest: str = ""
    population_digest: str = ""


class PopulationSummary(Contract):
    """Deterministic summary projected from the canonical population rows."""

    unit_count: int = Field(ge=1)
    unit_type: str = Field(min_length=1)
    labeled_count: int = Field(default=0, ge=0)
    unlabeled_count: int = Field(default=0, ge=0)
    source_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.labeled_count + self.unlabeled_count > self.unit_count:
            raise ValueError("labeled and unlabeled population counts exceed unit_count")
        if any(not key or value < 0 for key, value in self.source_counts.items()):
            raise ValueError("source_counts keys must be non-empty and values non-negative")
        if self.source_counts and sum(self.source_counts.values()) != self.unit_count:
            raise ValueError("source_counts must sum to unit_count")
        return self


class SourceReport(Contract):
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    population: PopulationSummary
    limitations: list[Any] = Field(default_factory=list)


class PreparationMapping(Contract):
    population_policy: PopulationPolicy
    case_identity: CaseIdentityRule
    grouping: GroupingRule
    joins: list[JoinRule] = Field(default_factory=list)
    fields: list[FieldRule] = Field(min_length=1)
    asset_rules: list[AssetRule] = Field(default_factory=list)
    target_path_rule: Literal["stable_identity"] = "stable_identity"
    field_semantics: dict[str, Any] = Field(default_factory=dict)
    resolved_decision_refs: list[str] = Field(default_factory=list)
    limitations: list[Any] = Field(default_factory=list)


class ExecutionReport(Contract):
    mapping_digest: str = ""
    population_digest: str = ""
    input_digest: str = ""
    tests: list[dict[str, Any]] = Field(default_factory=list)
    full_execution: dict[str, Any] = Field(default_factory=dict)
    failed_checks: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[Any] = Field(default_factory=list)
    population_accounting: dict[str, Any] = Field(default_factory=dict)
    mapping_rule_coverage: dict[str, Any] = Field(default_factory=dict)


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
    """The small host-owned semantic boundary between the two sessions."""

    handoff_id: str = ""
    dataset_id: str = Field(min_length=1)
    source_snapshot_digest: str = Field(min_length=1)
    source_report: ArtifactRef
    mapping: ArtifactRef
    population: ArtifactRef
    inspection_script: ArtifactRef
    image_conversion_plan: ArtifactRef
    decisions: list[UserDecision] = Field(default_factory=list)
    validation_receipt_ref: str = Field(min_length=1)
    digest: str = Field(min_length=1)


class ValidationIssue(Contract):
    issue_id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    stage: Stage
    message: str = Field(min_length=1)
    severity: Literal["error", "warning"] = "error"
    rule_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    record_ref: str | None = None
    path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    expected: str = ""
    observed: str = ""


class ValidationResult(Contract):
    status: Literal["accepted", "rejected", "needs_input", "blocked"]
    stage: Stage
    control: Literal["mapping_revision_required"] | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    questions: list[SemanticQuestion] = Field(default_factory=list)
    accepted_handoff: AcceptedHandoff | None = None
    valid_case_ids: list[str] = Field(default_factory=list)
    quarantined_cases: list[dict[str, Any]] = Field(default_factory=list)
    dropped_evidence: list[dict[str, Any]] = Field(default_factory=list)
    dependency_refs: list[str] = Field(default_factory=list)
    accounting: dict[str, int] = Field(default_factory=dict)
    receipt_ref: str | None = None
    report: dict[str, Any] = Field(default_factory=dict)
    candidate_digest: str = ""
    digest_ledger: dict[str, DigestEntry] = Field(default_factory=dict)


class StageOutputs(Contract):
    root: str = "outputs"
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    contracts_root: str = "inputs/contracts"
    files: list[dict[str, Any]] = Field(default_factory=list)


class ContextPacket(Contract):
    """The bounded serialisable projection sent to one coding session."""

    identity: dict[str, Any]
    task: dict[str, Any]
    sources: list[SourceBinding] = Field(default_factory=list)
    accepted_handoff: AcceptedHandoff | None = None
    decisions: list[UserDecision] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    files: list[InputBinding] = Field(default_factory=list)
    outputs: StageOutputs
    validation_feedback: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    budgets: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self):
        required = {"run_id", "dataset_id", "stage", "attempt", "input_digest"}
        missing = required - set(self.identity)
        if missing:
            raise ValueError(f"Context identity is missing: {sorted(missing)}")
        if self.identity["stage"] not in {item.value for item in Stage}:
            raise ValueError("Context identity contains an unknown stage")
        return self


class Submission(Contract):
    """Worker's explicit candidate handoff; it is not host acceptance."""

    requested_outcome: Literal["submitted", "needs_input", "blocked"]
    summary: str = ""
    outputs: list[str] = Field(default_factory=list)
    question_refs: list[str] = Field(default_factory=list)
    diagnostic_codes: list[str] = Field(default_factory=list)


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
    source_roots: dict[str, str] = Field(default_factory=dict)
    source_manifests: dict[str, str] = Field(default_factory=dict)
    input_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    input_digest: str = Field(min_length=1)
    source_snapshot_digest: str = ""
    prompt_token_count: int | None = Field(default=None, ge=0)
    required_outputs: list[str] = Field(default_factory=list)
    optional_outputs: list[str] = Field(default_factory=list)
    contracts_root: str = "inputs/contracts"
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


class StageSpec(Contract):
    stage: Stage
    objective: str = Field(min_length=1)
    required_outputs: list[str] = Field(min_length=1)
    optional_outputs: list[str] = Field(default_factory=list)
    input_roles: list[str] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)
    commands: list[list[str]] = Field(default_factory=list)
    completion_by_outcome: dict[str, list[str]] = Field(default_factory=dict)


STAGE_SPECS: dict[str, StageSpec] = {
    Stage.UNDERSTAND.value: StageSpec(
        stage=Stage.UNDERSTAND,
        objective="Understand source semantics once; freeze mapping and image conversion policy.",
        required_outputs=["mapping.json", "image_conversion_plan.json", "source_report.json",
                          "image_population.jsonl", "inspect_source.py", "semantic_questions.json", "submission.json"],
        commands=[["python", "-m", "medbenchagent.preparation.cli", "validate-understand",
                   "--context", "inputs/context.json", "--candidate", "outputs", "--report", "work/validation.json"]],
    ),
    Stage.PREPARE_TEXT.value: StageSpec(
        stage=Stage.PREPARE_TEXT,
        objective="Implement accepted annotation mapping without converting images or reinterpreting semantics.",
        required_outputs=["cases.jsonl", "provenance.jsonl", "image_requests.jsonl", "submission.json"],
        optional_outputs=["sampling_manifest.jsonl", "split_manifest.jsonl", "semantic_questions.json"],
    ),
    Stage.PREPARE_IMAGE.value: StageSpec(
        stage=Stage.PREPARE_IMAGE,
        objective="Materialize requested presentations and mask features under accepted conversion policy.",
        required_outputs=["asset_manifest.jsonl", "presentation_manifest.jsonl", "mask_features.jsonl", "submission.json"],
        optional_outputs=["semantic_questions.json"],
    ),
}

CONTRACT_RULES = """# Preparation contracts

Use the supplied schemas and shared validators. Source locators remain inside frozen roots.
Case/Evidence identities and relations come from accepted mapping. Source dispositions use
case_ids and record_refs, one terminal outcome per source unit. Images use stable target paths
from medbenchagent.data.presentation.target_path. Existing JPEG/PNG bytes are preserved when
no transform is requested; medical presentations are actual PNG. Reference masks stay hidden.
Understand must not predict conversion results. Text declares requests; Image records facts.
Implementations and diagnostics belong in work/. Only Host packages and publishes.
"""


def contract_file_names(stage: Stage | str) -> list[str]:
    common = ["rules.md", "submission.schema.json", "question.schema.json"]
    if Stage(stage) is Stage.UNDERSTAND:
        return common + ["source_report.schema.json", "population.schema.json", "mapping.schema.json",
                         "image_conversion_plan.schema.json"]
    return common + ["case.schema.json", "asset.schema.json", "presentation.schema.json",
                     "split.schema.json", "sampling.schema.json", "image_request.schema.json",
                     "population_outcome.schema.json"]


class ImageConversionPlan(Contract):
    policies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ImageRequest(Contract):
    case_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_format: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    view_id: str | None = None
    panel_id: str | None = None
    order: int = Field(ge=0)
    target_path: str = Field(min_length=1)
    reference_only: bool = False


def digest_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
