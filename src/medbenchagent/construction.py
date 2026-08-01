"""Append-only construction events and their compact derived report."""

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

SPECIFICATION_AGENT = "SpecificationAgent"
CAPABILITY_AGENT = "CapabilityAgent"
CLINICAL_DESIGN_AGENT = "ClinicalDesignAgent"
GENERATION_AGENT = "GenerationAgent"
AUDIT_AGENT = "AuditAgent"
AGENT_NAMES = (
    SPECIFICATION_AGENT,
    CAPABILITY_AGENT,
    CLINICAL_DESIGN_AGENT,
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
    )


def construction_report(events_path: str | Path) -> dict[str, object]:
    events = [json.loads(line) for line in Path(events_path).read_text(encoding="utf-8").splitlines() if line]
    llm = [event for event in events if event.get("event_type") == "llm_call"]
    started = next((event for event in events if event.get("event_type") == "run_started"), {})
    completed = next((event for event in reversed(events) if event.get("event_type") == "run_completed"), {})
    elapsed = None
    if isinstance(started.get("timestamp"), str) and isinstance(completed.get("timestamp"), str):
        elapsed = round(
            (
                datetime.fromisoformat(completed["timestamp"].replace("Z", "+00:00"))
                - datetime.fromisoformat(started["timestamp"].replace("Z", "+00:00"))
            ).total_seconds(),
            3,
        )
    summary = next((event for event in reversed(events) if event.get("event_type") == "generation_summary"), {})
    task_counts = summary.get("task_counts", {}) if isinstance(summary.get("task_counts"), dict) else {}
    shortfalls = summary.get("shortfalls", {}) if isinstance(summary.get("shortfalls"), dict) else {}
    raw_items = sum(value for value in task_counts.values() if isinstance(value, int))
    shortfall = sum(value for value in shortfalls.values() if isinstance(value, int))
    target = raw_items + shortfall
    audit_summary = next(
        (event for event in reversed(events) if event.get("event_type") == "audit_summary"), {}
    )
    audited = audit_summary.get("audited", raw_items)
    rejected = audit_summary.get("rejected", 0)
    if not isinstance(audited, int) or audited < 0:
        audited = raw_items
    if not isinstance(rejected, int) or rejected < 0:
        rejected = 0
    agent_metrics = {
        name: {
            "calls": sum(event.get("agent") == name for event in llm),
            "input_tokens": sum(int(event.get("input_tokens", 0)) for event in llm if event.get("agent") == name),
            "output_tokens": sum(int(event.get("output_tokens", 0)) for event in llm if event.get("agent") == name),
            "latency_ms": sum(float(event.get("latency_ms", 0)) for event in llm if event.get("agent") == name),
        }
        for name in AGENT_NAMES
    }
    human_stages = {
        stage: {"confirmed": 0, "revision_rounds": 0, "modified_objects": 0}
        for stage in ("benchmark_info", "dataset", "task_plan", "sample_count", "materials", "rag_design")
    }
    human_stages["materials"]["selected_objects"] = 0
    for event in events:
        if event.get("stage") not in human_stages:
            continue
        stage = human_stages[event["stage"]]
        if (
            event.get("event_type") == "human_interaction"
            and event.get("parse_status") == "success"
            and event.get("action") == "confirm"
        ):
            stage["confirmed"] = 1
            selected = event.get("selected_objects")
            if event.get("stage") == "materials" and isinstance(selected, int) and selected >= 0:
                stage["selected_objects"] = selected
        elif event.get("event_type") == "revision_applied":
            stage["revision_rounds"] += 1
            modified = event.get("modified_objects", 0)
            if isinstance(modified, int) and modified >= 0:
                stage["modified_objects"] += modified
    input_tokens = sum(int(event.get("input_tokens", 0)) for event in llm)
    output_tokens = sum(int(event.get("output_tokens", 0)) for event in llm)
    return {
        "schema_version": "construction-report.v2",
        "wall_clock_seconds": elapsed,
        "llm": {
            "calls": len(llm),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": sum(float(event.get("latency_ms", 0)) for event in llm),
            "per_audited_item": {
                "calls": len(llm) / audited if audited else None,
                "tokens": (input_tokens + output_tokens) / audited if audited else None,
            },
        },
        "items": {
            "target": target,
            "raw": raw_items,
            "audited": audited,
            "rejected": rejected,
            "shortfall": shortfall,
        },
        "agents": agent_metrics,
        "human_stages": human_stages,
    }
