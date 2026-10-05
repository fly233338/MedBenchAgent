"""Run-scoped state, memory and context contracts for data preparation."""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medbenchagent.runtime.run_artifacts import atomic_write_json


class PreparationPhase(StrEnum):
    BOOTSTRAP = "bootstrap"
    INVENTORY = "inventory"
    SEMANTIC = "semantic"
    ADAPTER = "adapter"
    CANDIDATE = "candidate"
    VALIDATION = "validation"
    REPAIR = "repair"


class PreparationStatus(StrEnum):
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    FAILED = "failed"
    BLOCKED = "blocked"
    COMPLETE = "complete"


class TerminationReason(StrEnum):
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    NO_PROGRESS = "no_progress"
    TOOL_ERROR_BUDGET = "tool_error_budget"
    ITERATION_BUDGET = "iteration_budget"
    TIME_BUDGET = "time_budget"
    WORKER_ERROR = "worker_error"
    SOURCE_CHANGED = "source_changed"
    VALIDATION_FAILED = "validation_failed"
    OUTPUT_INVALID = "output_invalid"
    CANCELLED = "cancelled"
    CONTEXT_LIMIT = "context_limit"


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: str = "candidate"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    value: Any
    producer: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    status: str = "proposal"
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    digest: str = ""

    @model_validator(mode="after")
    def content_digest(self):
        payload = json.dumps(self.value, ensure_ascii=False, sort_keys=True, default=str)
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self.digest and self.digest != expected:
            raise ValueError("Memory record digest does not match its value")
        self.digest = expected
        return self


class MemoryState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "preparation-memory"
    verified_facts: list[MemoryRecord] = Field(default_factory=list)
    decisions: list[MemoryRecord] = Field(default_factory=list)
    open_questions: list[MemoryRecord] = Field(default_factory=list)
    constraints: list[MemoryRecord] = Field(default_factory=list)
    validator_feedback: list[MemoryRecord] = Field(default_factory=list)
    next_actions: list[MemoryRecord] = Field(default_factory=list)


class ProgressState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "preparation-progress"
    iterations: int = Field(default=0, ge=0)
    tool_errors: int = Field(default=0, ge=0)
    empty_responses: int = Field(default=0, ge=0)
    no_progress: int = Field(default=0, ge=0)
    last_action: str = ""
    last_progress_at: str = ""
    artifact_digest: str = ""


class PreparationState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "preparation-state"
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    source_digest: str = ""
    input_digest: str = ""
    phase: PreparationPhase = PreparationPhase.BOOTSTRAP
    status: PreparationStatus = PreparationStatus.RUNNING
    attempt: int = Field(default=1, ge=1)
    session_id: str = ""
    termination_reason: TerminationReason | None = None
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    progress: ProgressState = Field(default_factory=ProgressState)
    pending_issue_id: str = ""
    logs: dict[str, str] = Field(default_factory=dict)
    partial_outputs: list[str] = Field(default_factory=list)
    missing_outputs: list[str] = Field(default_factory=list)

    def next_session(self, phase: PreparationPhase, attempt: int | None = None) -> "PreparationState":
        number = attempt or (self.attempt + 1 if phase == self.phase else 1)
        session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.run_id}:{self.dataset_id}:{phase}:{number}"))
        return self.model_copy(update={
            "phase": phase,
            "status": PreparationStatus.RUNNING,
            "attempt": number,
            "session_id": session,
            "termination_reason": None,
            "pending_issue_id": "",
            "progress": ProgressState(),
        })


class StageBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    max_iterations: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)
    no_progress_limit: int = Field(default=8, ge=1)
    tool_error_limit: int = Field(default=3, ge=1)


class StageContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "preparation-context"
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    phase: PreparationPhase
    attempt: int = Field(ge=1)
    objective: str = Field(min_length=1)
    inputs: dict[str, ArtifactRef] = Field(default_factory=dict)
    memory: MemoryState = Field(default_factory=MemoryState)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    validator_feedback: list[dict[str, Any]] = Field(default_factory=list)
    allowed_outputs: list[str] = Field(default_factory=list)
    budget: StageBudget
    workspace: str = Field(min_length=1)


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: str = "preparation-stage-result"
    phase: PreparationPhase
    attempt: int = Field(ge=1)
    session_id: str = Field(min_length=1)
    status: PreparationStatus
    outputs: dict[str, str] = Field(default_factory=dict)
    question: str = ""
    diagnostic: str = ""
    termination_reason: TerminationReason | None = None
    progress: ProgressState = Field(default_factory=ProgressState)


def preparation_paths(workspace: str | Path) -> dict[str, Path]:
    root = Path(workspace)
    state = root / "state"
    return {
        "workflow": state / "workflow.json",
        "memory": state / "memory.json",
        "progress": state / "progress.json",
        "artifacts": state / "artifacts.json",
        "issues": state / "issues.json",
    }


def save_preparation_state(workspace: str | Path, state: PreparationState) -> None:
    atomic_write_json(preparation_paths(workspace)["workflow"], state.model_dump(mode="json"))


def load_preparation_state(workspace: str | Path) -> PreparationState | None:
    path = preparation_paths(workspace)["workflow"]
    if not path.is_file():
        return None
    return PreparationState.model_validate_json(path.read_text(encoding="utf-8"))


def save_memory(workspace: str | Path, memory: MemoryState) -> None:
    atomic_write_json(preparation_paths(workspace)["memory"], memory.model_dump(mode="json"))


def load_memory(workspace: str | Path) -> MemoryState:
    path = preparation_paths(workspace)["memory"]
    if not path.is_file():
        return MemoryState()
    return MemoryState.model_validate_json(path.read_text(encoding="utf-8"))


def save_progress(workspace: str | Path, progress: ProgressState) -> None:
    atomic_write_json(preparation_paths(workspace)["progress"], progress.model_dump(mode="json"))

