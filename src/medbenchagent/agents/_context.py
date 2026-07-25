"""Construction-event context shared by role-level graph wrappers."""

from __future__ import annotations

import os
from typing import Any

from medbenchagent.construction import event_context, record_human_action


def llm_context(state: dict[str, Any], agent: str, operation: str):
    return event_context(
        construction_events_path=state.get("construction_events_path"),
        run_id=state.get("run_id"),
        dataset_id=state.get("selected_dataset_id") or None,
        agent=agent,
        operation=operation,
        model=os.getenv("DEEPSEEK_MODEL"),
    )


def human_action(
    state: dict[str, Any], *, agent: str, stage: str, action: str, modified_objects: int
) -> None:
    if not state.get("construction_events_path") or not state.get("run_id"):
        return
    record_human_action(
        state["construction_events_path"],
        run_id=state["run_id"],
        dataset_id=state.get("selected_dataset_id") or None,
        agent=agent,
        stage=stage,
        action=action,
        modified_objects=modified_objects,
    )
