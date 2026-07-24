"""Append-only construction events and their compact derived report."""

from __future__ import annotations

import contextlib
import contextvars
import json
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("construction_context", default=None)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def record_event(path: str | Path, event_type: str, **values: object) -> dict[str, object]:
    """Append metadata only; callers must never pass prompts or model bodies."""
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
        status=status,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        input_tokens=prompt_tokens if available else 0,
        output_tokens=completion_tokens if available else 0,
        token_usage_available=available,
        error_type=error_type,
        call_id=call_id,
        provider_response_id=getattr(response, "id", None),
    )


def construction_report(events_path: str | Path) -> dict[str, object]:
    events = [json.loads(line) for line in Path(events_path).read_text(encoding="utf-8").splitlines() if line]
    llm = [event for event in events if event.get("event_type") == "llm_call"]
    accepted = [event for event in events if event.get("event_type") == "review_accept"]
    timestamps = [event["timestamp"] for event in events if isinstance(event.get("timestamp"), str)]
    elapsed = None
    if len(timestamps) >= 2:
        elapsed = round((datetime.fromisoformat(max(timestamps).replace("Z", "+00:00")) - datetime.fromisoformat(min(timestamps).replace("Z", "+00:00"))).total_seconds(), 3)
    counts = Counter(str(event.get("event_type")) for event in events)
    complete_usage = all(event.get("token_usage_available") is True for event in llm)
    accepted_count = len(accepted)
    return {
        "schema_version": "construction-report.v1",
        "wall_clock_seconds": elapsed,
        "llm_calls": len(llm),
        "input_tokens": sum(int(event.get("input_tokens", 0)) for event in llm),
        "output_tokens": sum(int(event.get("output_tokens", 0)) for event in llm),
        "token_usage_complete": complete_usage,
        "accepted_item_averages": {
            "llm_calls": (len(llm) / accepted_count) if accepted_count else None,
            "tokens": ((sum(int(event.get("input_tokens", 0)) + int(event.get("output_tokens", 0)) for event in llm)) / accepted_count) if accepted_count else None,
            "llm_latency_ms": (sum(float(event.get("latency_ms", 0)) for event in llm) / accepted_count) if accepted_count else None,
        },
        "counts": dict(counts),
        "retention": 1.0 if accepted_count else None,
    }
