"""Construction-event context shared by role-level graph wrappers."""

from __future__ import annotations

from typing import Any

from medbenchagent.construction import (
    event_context,
    record_artifact_snapshot,
    record_human_interaction,
    record_revision_applied,
)
from medbenchagent.settings import get_settings


def llm_context(state: dict[str, Any], agent: str, operation: str):
    return event_context(
        construction_events_path=state.get("construction_events_path"),
        run_id=state.get("run_id"),
        dataset_id=state.get("selected_dataset_id") or None,
        agent=agent,
        operation=operation,
        model=get_settings().llm.model,
    )


def human_interaction(
    state: dict[str, Any],
    *,
    agent: str,
    stage: str,
    user_input: object,
    parse_status: str,
    action: str | None = None,
    error_type: str | None = None,
    selected_objects: int | None = None,
) -> str:
    interaction_id = ""
    if not state.get("construction_events_path") or not state.get("run_id"):
        return interaction_id
    return record_human_interaction(
        state["construction_events_path"],
        run_id=state["run_id"],
        dataset_id=state.get("selected_dataset_id") or None,
        agent=agent,
        stage=stage,
        user_input=user_input if isinstance(user_input, str) else str(user_input),
        parse_status=parse_status,
        action=action,
        error_type=error_type,
        selected_objects=selected_objects,
    )


def revision_applied(
    state: dict[str, Any],
    *,
    agent: str,
    stage: str,
    interaction_id: str,
    modified_object_ids: list[str],
    artifact_type: str | None = None,
    artifact: dict[str, str | int] | None = None,
) -> None:
    if not interaction_id or not state.get("construction_events_path") or not state.get("run_id"):
        return
    record_revision_applied(
        state["construction_events_path"],
        run_id=state["run_id"],
        dataset_id=state.get("selected_dataset_id") or None,
        agent=agent,
        stage=stage,
        interaction_id=interaction_id,
        modified_object_ids=modified_object_ids,
        artifact_type=artifact_type,
        artifact_version=artifact["version"] if artifact else None,
        artifact_path=artifact["relative_path"] if artifact else None,
    )


def artifact_snapshot(
    state: dict[str, Any],
    *,
    agent: str,
    stage: str,
    artifact_type: str,
    artifact: dict[str, str | int],
) -> None:
    if not state.get("construction_events_path") or not state.get("run_id"):
        return
    record_artifact_snapshot(
        state["construction_events_path"],
        run_id=state["run_id"],
        dataset_id=state.get("selected_dataset_id") or None,
        agent=agent,
        stage=stage,
        artifact_type=artifact_type,
        artifact=artifact,
        source="initial_model_output",
    )
