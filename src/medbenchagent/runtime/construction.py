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

_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "construction_context", default=None
)

def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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


def record_llm_call(
    *, status: str, started: float, error_type: str | None = None, response: object = None
) -> None:
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
        source_id=context.get("source_id"),
        task_id=context.get("task_id"),
        item_id=context.get("item_id"),
        agent=context.get("agent"),
        operation=context.get("operation", "llm"),
        component=context.get("component"),
        model=context.get("model") or get_settings().llm.model,
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        input_tokens=prompt_tokens if available else 0,
        output_tokens=completion_tokens if available else 0,
        context_tokens=prompt_tokens if available else None,
        context_window_tokens=context.get("context_window_tokens"),
        context_ratio=(prompt_tokens / context.get("context_window_tokens")
                       if available and context.get("context_window_tokens") else None),
        reasoning_effort=context.get("reasoning_effort", get_settings().llm.reasoning_effort) or None,
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
        source_id=context.get("source_id"),
        task_id=context.get("task_id"),
        agent=context.get("agent"),
        operation=context.get("operation", "llm"),
        component=context.get("component"),
        model=context.get("model") or get_settings().llm.model,
        validation_errors=errors,
        model_output=model_output,
        call_id=call_id,
    )
