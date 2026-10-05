"""Typed private-state boundaries; tools keep their existing dictionary-based updates."""

from pathlib import Path
from typing import Any, Literal, TypedDict
from pydantic import BaseModel, ConfigDict, Field
from medbenchagent.runtime.run_artifacts import atomic_write_json


class SupervisorContext(TypedDict, total=False):
    run_directory: str
    operation_directory: str
    bir_path: str
    bir_revision: int
    bir_manifest_digest: str
    bir_template_revision: str
    bir_summary_ref: dict[str, Any]
    artifact_versions: dict[str, str]
    child_states: dict[str, dict[str, str]]
    issues: dict[str, dict]
    evidence_index_path: str


class TaskPlanningContext(SupervisorContext, total=False):
    dataset_scope: str
    dataset_access_manifest: dict
    draft_path: str


class DataProcessingContext(SupervisorContext, total=False):
    dataset_scope: str
    source_scope: str | None
    bundle_root: str
    sources: list[dict[str, Any]]
    source_roots: dict[str, str]
    source_manifests: dict[str, str]
    preparation_workspace: str | None
    preparation_candidate: dict[str, Any] | None
    preparation_population_path: str | None
    preparation_mapping_digest: str | None


class AllocationContext(SupervisorContext, total=False):
    selected_dataset_ids: list[str]
    allocation_preferences: list[dict]
    allocation_path: str
    allocation: dict[str, Any]


class FinalizationContext(AllocationContext, total=False):
    dataset_checks: dict
    delivery_path: str
    dataset_generations: dict[str, dict[str, Any]]
    delivery_counts: dict[str, Any]


class ContextBoundary(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    run_directory: str
    operation_directory: str
    artifact_versions: dict[str, str] = Field(default_factory=dict)
    dataset_scope: str | None = None
    selected_dataset_id: str | None = None
    source_scope: str | None = None
    evidence_index_path: str | None = None
    child_states: dict[str, dict[str, str]] = Field(default_factory=dict)
    issues: dict[str, dict] = Field(default_factory=dict)
    decisions: dict[str, str] = Field(default_factory=dict)


class PrivateState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["agent-private-state"] = "agent-private-state"
    role: Literal["TaskPlanningAgent", "DataProcessingAgent", "AllocationAgent", "FinalizationAgent", "GenerationAgent"]
    dataset_id: str | None
    context: dict[str, Any]


ROLES = {"planning": "TaskPlanningAgent", "preparation": "DataProcessingAgent", "templates": "TaskPlanningAgent",
         "generation": "GenerationAgent", "allocation": "AllocationAgent", "finalization": "FinalizationAgent"}


def validate_context(context, role, dataset):
    ContextBoundary.model_validate(context)
    if context.get("dataset_scope", context.get("selected_dataset_id")) != dataset:
        raise ValueError("Private context belongs to another dataset")
    if context.get("agent_role", role) != role:
        raise ValueError("Private context belongs to another role")
    for key, value in context.items():
        if key.endswith("_path") and value is not None and not isinstance(value, str):
            raise ValueError(f"Private state {key} must be a file reference")
    if any(key in context for key in ("inspections", "source_documents", "retrieved_passages", "retrieved_evidence")):
        raise ValueError("Private state must reference evidence instead of embedding payloads")
    if role in {"TaskPlanningAgent", "DataProcessingAgent", "GenerationAgent"} and not dataset:
        raise ValueError("Dataset specialist state requires a dataset identity")
    if role in {"AllocationAgent", "FinalizationAgent"} and dataset is not None:
        raise ValueError("Global specialist state cannot carry a dataset scope")
    if context.get("evidence_index_path"):
        expected = Path(context["run_directory"]) / "evidence/index.sqlite"
        if Path(context["evidence_index_path"]).resolve() != expected.resolve():
            raise ValueError("Evidence index must belong to the current run")


def save_context(path, context, role, dataset=None):
    role = ROLES.get(role, role)
    context = {**context, "agent_role": role}
    validate_context(context, role, dataset)
    atomic_write_json(path, PrivateState(role=role, dataset_id=dataset, context=context).model_dump())


def load_context(path, role, dataset=None):
    role = ROLES.get(role, role)
    state = PrivateState.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if state.role != role or state.dataset_id != dataset:
        raise ValueError("Saved private state role/dataset does not match its delegation")
    validate_context(state.context, role, dataset)
    return state.context

