"""Append-only construction events for model calls and tool execution."""

from __future__ import annotations

import contextlib
import contextvars
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from medbenchagent.settings import get_settings

_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("construction_context", default=None)

SUPERVISOR_AGENT = "SupervisorAgent"
TASK_PLANNING_AGENT = "TaskPlanningAgent"
DATA_PROCESSING_AGENT = "DataProcessingAgent"
GENERATION_AGENT = "GenerationAgent"
AUDIT_AGENT = "AuditAgent"
AGENT_NAMES = (
    SUPERVISOR_AGENT,
    TASK_PLANNING_AGENT,
    DATA_PROCESSING_AGENT,
    GENERATION_AGENT,
    AUDIT_AGENT,
)

_MISSING = object()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def changed_mapping_keys(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Return stable string keys whose values differ between two mappings."""
    return sorted(
        key
        for key in before.keys() | after.keys()
        if before.get(key, _MISSING) != after.get(key, _MISSING)
    )


def record_event(path: str | Path, event_type: str, **values: object) -> dict[str, object]:
    """Append one construction event as compact UTF-8 JSON."""
    event = {"timestamp": _timestamp(), "event_type": event_type} | {
        key: value for key, value in values.items() if value is not None
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    return event


@contextlib.contextmanager
def event_context(**values: object) -> Iterator[list[str]]:
    """Associate nested model calls with one workflow operation and collect local IDs."""
    parent = _CONTEXT.get() or {}
    call_ids: list[str] = []
    token = _CONTEXT.set(parent | values | {"call_ids": call_ids})
    try:
        yield call_ids
    finally:
        _CONTEXT.reset(token)


@contextlib.contextmanager
def event_operation(operation: str) -> Iterator[None]:
    context = _CONTEXT.get()
    if context is None:
        yield
        return
    token = _CONTEXT.set(context | {"operation": operation})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def record_llm_call(*, status: str, started: float, error_type: str | None = None, response: object = None) -> None:
    context = _CONTEXT.get()
    if context is None or not context.get("construction_events_path"):
        return
    call_id = str(uuid.uuid4())
    context["call_ids"].append(call_id)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    available = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
    record_event(
        str(context["construction_events_path"]),
        "llm_call",
        run_id=context.get("run_id"),
        dataset_id=context.get("dataset_id"),
        task_id=context.get("task_id"),
        item_id=context.get("item_id"),
        agent=context.get("agent", "deepseek"),
        operation=context.get("operation", "llm"),
        component=context.get("component"),
        model=context.get("model") or get_settings().llm.model,
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        input_tokens=prompt_tokens if available else 0,
        output_tokens=completion_tokens if available else 0,
        token_usage_available=available,
        error_type=error_type,
        call_id=call_id,
        provider_response_id=getattr(response, "id", None),
    )


def record_llm_validation_error(*, errors: list[str], model_output: object) -> None:
    """Record a model response that failed deterministic contract validation."""
    context = _CONTEXT.get()
    if context is None or not context.get("construction_events_path"):
        return
    call_ids = context.get("call_ids")
    call_id = call_ids[-1] if isinstance(call_ids, list) and call_ids else None
    record_event(
        str(context["construction_events_path"]),
        "llm_validation_error",
        run_id=context.get("run_id"),
        dataset_id=context.get("dataset_id"),
        task_id=context.get("task_id"),
        agent=context.get("agent", "deepseek"),
        operation=context.get("operation", "llm"),
        component=context.get("component"),
        model=context.get("model") or get_settings().llm.model,
        validation_errors=errors,
        model_output=model_output,
        call_id=call_id,
    )


def record_human_interaction(
    path: str | Path,
    *,
    run_id: str,
    dataset_id: str | None,
    agent: str,
    stage: str,
    user_input: str,
    parse_status: str,
    action: str | None = None,
    error_type: str | None = None,
    selected_objects: int | None = None,
) -> str:
    """Record one natural-language interaction, including failed parse attempts."""
    interaction_id = str(uuid.uuid4())
    record_event(
        path,
        "human_interaction",
        run_id=run_id,
        dataset_id=dataset_id,
        agent=agent,
        stage=stage,
        interaction_id=interaction_id,
        user_input=user_input,
        parse_status=parse_status,
        action=action,
        error_type=error_type,
        selected_objects=selected_objects,
    )
    return interaction_id


def record_revision_applied(
    path: str | Path,
    *,
    run_id: str,
    dataset_id: str | None,
    agent: str,
    stage: str,
    interaction_id: str,
    modified_object_ids: list[str],
    artifact_type: str | None = None,
    artifact_version: int | None = None,
    artifact_path: str | None = None,
) -> None:
    """Record one successfully completed revision and its actual changed objects."""
    record_event(
        path,
        "revision_applied",
        run_id=run_id,
        dataset_id=dataset_id,
        agent=agent,
        stage=stage,
        interaction_id=interaction_id,
        modified_objects=len(modified_object_ids),
        modified_object_ids=modified_object_ids,
        artifact_type=artifact_type,
        artifact_version=artifact_version,
        artifact_path=artifact_path,
    )


def record_artifact_snapshot(
    path: str | Path,
    *,
    run_id: str,
    dataset_id: str | None,
    agent: str,
    stage: str,
    artifact_type: str,
    artifact: dict[str, str | int],
    source: str,
) -> None:
    """Record the first immutable artifact snapshot created by a model stage."""
    record_event(
        path,
        "artifact_snapshot",
        run_id=run_id,
        dataset_id=dataset_id,
        agent=agent,
        stage=stage,
        artifact_type=artifact_type,
        artifact_version=artifact["version"],
        artifact_path=artifact["relative_path"],
        source=source,
    )


