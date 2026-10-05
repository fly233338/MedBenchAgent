"""Canonical contracts for the two coding stages.

These contracts are intentionally independent of an OpenHands SDK model.  The
worker receives a :class:`StageRequest`, writes a machine readable submission,
and returns a :class:`StageResult`; host validation decides whether files are
accepted.  No conversation summary or general purpose memory is part of the
contract.
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
    PREPARE = "prepare"


class AttemptMode(StrEnum):
    INITIAL = "initial"
    REPAIR = "repair"
    CLARIFICATION = "clarification"
    CONTINUE = "continue"


ATTEMPT_MODES = tuple(item.value for item in AttemptMode)


class ArtifactRef(Contract):
    """A host-addressable file accepted or produced by a stage."""

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: Literal["candidate", "accepted", "invalid", "diagnostic"] = "candidate"
    depends_on: list[str] = Field(default_factory=list)
    purpose: str = ""


class InputBinding(Contract):
    """A file the worker can read in its workspace."""

    path: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    role: str = Field(min_length=1)
    read_policy: Literal["before_work", "stream_during_execution", "on_failure", "optional"] = "before_work"
    required: bool = True
    reason: str = ""


class SourceBinding(Contract):
    source_id: str = Field(min_length=1)
    root: str = Field(min_length=1)
    manifest: str = Field(min_length=1)
    digest: str = ""
    read_only: bool = True


class StageBudget(Contract):
    max_iterations: int = Field(default=40, ge=1)
    timeout_seconds: int = Field(default=1800, ge=1)
    no_progress_limit: int = Field(default=8, ge=1)
    tool_error_limit: int = Field(default=3, ge=1)
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


class SemanticQuestion(Contract):
    """A source-scoped question that can pause a stage."""

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


class ArtifactDependency(Contract):
    path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    digest: str = ""


class OutputFile(Contract):
    path: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    purpose: str = ""
    depends_on: list[str] = Field(default_factory=list)


class OutputManifest(Contract):
    files: list[OutputFile] = Field(default_factory=list)
    dependencies: list[ArtifactDependency] = Field(default_factory=list)
    mapping_digest: str = ""
    population_digest: str = ""


class SourceReport(Contract):
    sources: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    population: dict[str, Any] = Field(default_factory=dict)
    limitations: list[Any] = Field(default_factory=list)


class PreparationMapping(Contract):
    population_policy: dict[str, Any] = Field(default_factory=dict)
    case_identity: dict[str, Any] = Field(default_factory=dict)
    grouping: dict[str, Any] = Field(default_factory=dict)
    joins: list[dict[str, Any]] = Field(default_factory=list)
    fields: list[dict[str, Any]] = Field(default_factory=list)
    asset_rules: list[dict[str, Any]] = Field(default_factory=list)
    field_semantics: dict[str, Any] = Field(default_factory=dict)
    resolved_decision_refs: list[str] = Field(default_factory=list)
    limitations: list[Any] = Field(default_factory=list)


class ExecutionReport(Contract):
    tests: list[dict[str, Any]] = Field(default_factory=list)
    full_execution: dict[str, Any] = Field(default_factory=dict)
    failed_checks: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[Any] = Field(default_factory=list)
    population_accounting: dict[str, Any] = Field(default_factory=dict)
    mapping_rule_coverage: dict[str, Any] = Field(default_factory=dict)


class UserDecision(Contract):
    decision_id: str = Field(min_length=1)
    issue_id: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    answer: str = Field(min_length=1)
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


class AttemptContext(Contract):
    mode: AttemptMode = AttemptMode.INITIAL
    target_issue_ids: list[str] = Field(default_factory=list)
    feedback_path: str | None = None
    previous_outputs_root: str | None = None
    resume_summary: str = ""
    completed_work: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)
    invalidated_work: list[str] = Field(default_factory=list)
    user_answer_refs: list[str] = Field(default_factory=list)


class StageOutputs(Contract):
    root: str = "outputs"
    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)
    contracts_root: str = "inputs/contracts"


class ContextPacket(Contract):
    """The bounded, serialisable projection sent to a coding worker."""

    identity: dict[str, Any]
    task: dict[str, Any]
    sources: list[SourceBinding] = Field(default_factory=list)
    bindings: dict[str, Any] = Field(default_factory=dict)
    decisions: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    files: list[InputBinding] = Field(default_factory=list)
    outputs: StageOutputs
    attempt: AttemptContext = Field(default_factory=AttemptContext)
    diagnostics: list[Diagnostic] = Field(default_factory=list)

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
    """Worker's explicit stage handoff; it is not host acceptance."""

    requested_outcome: Literal["submitted", "needs_input", "blocked"]
    summary: str = ""
    outputs: list[str] = Field(default_factory=list)
    question_refs: list[str] = Field(default_factory=list)
    diagnostic_codes: list[str] = Field(default_factory=list)
    completed_checks: list[str] = Field(default_factory=list)
    remaining_work: list[str] = Field(default_factory=list)


class StageRequest(Contract):
    stage: Stage
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    conversation_id: str = Field(min_length=1)
    execution_key: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    prompt_path: str = Field(min_length=1)
    context_path: str = Field(min_length=1)
    source_roots: dict[str, str] = Field(default_factory=dict)
    source_manifests: dict[str, str] = Field(default_factory=dict)
    input_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    input_digest: str = Field(min_length=1)
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

    @model_validator(mode="after")
    def check_paths(self):
        if not self.required_outputs:
            raise ValueError("A stage request must declare required outputs")
        return self


class StageResult(Contract):
    stage: Stage
    attempt: int = Field(ge=1)
    conversation_id: str = Field(min_length=1)
    execution_key: str = Field(min_length=1)
    status: Literal["submitted", "needs_input", "blocked", "failed"]
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


class StageSpec(Contract):
    stage: Stage
    objective: str = Field(min_length=1)
    required_outputs: list[str] = Field(min_length=1)
    optional_outputs: list[str] = Field(default_factory=list)
    input_roles: list[str] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)


STAGE_SPECS: dict[str, StageSpec] = {
    Stage.UNDERSTAND.value: StageSpec(
        stage=Stage.UNDERSTAND,
        objective="Understand every configured source and produce a source-grounded, implementable preparation mapping.",
        required_outputs=[
            "source_report.json", "image_population.jsonl", "mapping.json",
            "semantic_questions.json", "inspect_source.py", "submission.json",
        ],
        input_roles=["user_request", "source_index", "environment", "decisions", "contracts"],
        success_conditions=[
            "Every source unit is represented or explicitly accounted for.",
            "Mapping decisions have resolvable evidence references.",
            "Blocking questions are resolved before prepare starts.",
        ],
    ),
    Stage.PREPARE.value: StageSpec(
        stage=Stage.PREPARE,
        objective="Implement, test, execute and submit the accepted preparation mapping over the complete source population.",
        required_outputs=[
            "scripts/prepare.py", "cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl",
            "conversion_config.json", "grouping_policy.json", "execution_report.json",
            "output_manifest.json", "submission.json",
        ],
        optional_outputs=[
            "presentation_manifest.jsonl", "split_manifest.jsonl", "sampling_manifest.jsonl",
            "semantic_questions.json",
        ],
        input_roles=["user_request", "source_index", "environment", "decisions", "accepted", "validation"],
        success_conditions=[
            "Representative and boundary checks actually ran.",
            "The complete population is reconciled by stable source-unit identity.",
            "Every candidate dependency is registered in output_manifest.json.",
        ],
    ),
}


CONTRACT_RULES = """# Preparation contract rules

- Case IDs and Evidence IDs are unique within their scope.
- Asset Evidence values reference an Asset ID in asset_manifest.jsonl.
- Evidence refs stay within their Case and region Evidence references exactly one Asset Evidence.
- Region coordinates use normalized xyxy order; mask digests match their Asset digest.
- Source locators are source-relative and cannot escape the declared source root.
- Every emitted Case and every accounted source population unit has provenance.
- Population accounting distinguishes retained, isolated, excluded and duplicate units.
- Optional manifests reference existing Case, Asset and output files and preserve declared order.
- mapping, grouping_policy and conversion_config must carry the accepted mapping identity.
- output_manifest registers every submitted output and all file dependencies.
"""


def contract_file_names(stage: Stage | str) -> list[str]:
    """Return stable contract document names for a stage."""
    current = Stage(stage)
    common = ["rules.md", "submission.schema.json"]
    if current is Stage.UNDERSTAND:
        return common + ["source_report.schema.json", "mapping.schema.json", "question.schema.json"]
    return common + [
        "case.schema.json", "asset.schema.json", "presentation.schema.json", "split.schema.json",
        "sampling.schema.json", "execution_report.schema.json", "output_manifest.schema.json",
    ]


def digest_payload(value: Any) -> str:
    """Hash JSON data deterministically for execution and decision identities."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def digest_file(path: str | Path) -> str:
    """Hash a file without embedding host-specific paths in the digest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
