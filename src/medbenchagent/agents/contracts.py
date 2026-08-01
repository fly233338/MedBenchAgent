"""Role-level state contracts for the five benchmark-construction agents."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import MessagesState

from medbenchagent.construction import (
    AGENT_NAMES,
    AUDIT_AGENT,
    CAPABILITY_AGENT,
    CLINICAL_DESIGN_AGENT,
    GENERATION_AGENT,
    SPECIFICATION_AGENT,
)
from medbenchagent.contracts import GenerationRequest


class SpecificationAgentInputState(MessagesState, total=False):
    user_requirement: str
    benchmark_info: dict[str, Any]
    run_id: str
    construction_events_path: str
    working_blueprint_path: str


class SpecificationAgentState(SpecificationAgentInputState, total=False):
    benchmark_feedback: str
    benchmark_confirmed: bool
    initial_requirement_recorded: bool
    pending_benchmark_interaction_id: str
    pending_benchmark_info: dict[str, Any]


class SpecificationAgentOutputState(TypedDict):
    user_requirement: str
    benchmark_info: dict[str, Any]
    working_blueprint_path: str


class CapabilityAgentInputState(TypedDict):
    benchmark_info: dict[str, Any]
    working_blueprint_path: str
    run_id: str
    construction_events_path: str


class CapabilityAgentState(CapabilityAgentInputState, total=False):
    dataset_recommendation: dict[str, str]
    selected_dataset_id: str
    dataset_confirmed: bool
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    draft_task_plan: dict[str, Any]
    pending_task_feedback: dict[str, str] | None
    pending_task_interaction_id: str
    task_plan_confirmed: bool
    task_screening_report_path: str
    draft_sample_counts: dict[str, int]
    sample_counts_confirmed: bool
    selected_material_files: list[str]
    material_selection_error: str


class CapabilityAgentOutputState(TypedDict):
    working_blueprint_path: str
    selected_dataset_id: str
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    task_screening_report_path: str
    selected_material_files: list[str]


class ClinicalDesignAgentInputState(TypedDict):
    benchmark_info: dict[str, Any]
    working_blueprint_path: str
    selected_material_files: list[str]
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    run_id: str
    selected_dataset_id: str
    construction_events_path: str


class ClinicalDesignAgentState(ClinicalDesignAgentInputState, total=False):
    rag_task_queue: list[str]
    rag_task_feedback: dict[str, str]
    rag_current_task_id: str
    rag_completed_count: int
    rag_total_count: int
    designs_confirmed: bool


class ClinicalDesignAgentOutputState(TypedDict):
    working_blueprint_path: str


class GenerationAgentInputState(TypedDict, total=False):
    benchmark_info: dict[str, Any]
    run_id: str
    selected_dataset_id: str
    working_blueprint_path: str
    construction_events_path: str
    dataset_access_manifest: dict[str, Any]
    seed: int
    max_tasks_per_image: int
    question_temperature: float


class GenerationAgentState(GenerationAgentInputState, total=False):
    generation_request: GenerationRequest
    generation_target_counts: dict[str, int]
    generation_completed_counts: dict[str, int]
    generation_stage_index: int
    generation_stage: str
    generation_cursor: int
    generation_progress_path: str
    generation_rejections: dict[str, int]
    generation_exhausted: bool
    raw_output_path: str
    provenance_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


class GenerationAgentOutputState(TypedDict):
    raw_output_path: str
    provenance_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


class AuditAgentInputState(TypedDict, total=False):
    raw_output_path: str
    provenance_output_path: str
    working_blueprint_path: str
    selected_dataset_id: str
    construction_events_path: str
    run_directory: str
    run_id: str
    seed: int
    dataset_access_manifest: dict[str, Any]


class AuditAgentState(AuditAgentInputState, total=False):
    audit_artifacts: dict[str, str]


class AuditAgentOutputState(TypedDict):
    audit_artifacts: dict[str, str]


__all__ = [
    "AGENT_NAMES",
    "AUDIT_AGENT",
    "AuditAgentInputState",
    "AuditAgentOutputState",
    "AuditAgentState",
    "CAPABILITY_AGENT",
    "CLINICAL_DESIGN_AGENT",
    "CapabilityAgentInputState",
    "CapabilityAgentOutputState",
    "CapabilityAgentState",
    "ClinicalDesignAgentInputState",
    "ClinicalDesignAgentOutputState",
    "ClinicalDesignAgentState",
    "GENERATION_AGENT",
    "GenerationAgentInputState",
    "GenerationAgentOutputState",
    "GenerationAgentState",
    "SPECIFICATION_AGENT",
    "SpecificationAgentInputState",
    "SpecificationAgentOutputState",
    "SpecificationAgentState",
]
