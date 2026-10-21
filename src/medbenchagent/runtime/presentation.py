"""Shared user-facing projections for CLI and TUI event streams."""
import json


def compilation_text(run):
    view = run.get("view") or {}
    dataset = view.get("dataset_id") or run.get("dataset_id") or "unresolved"
    plan = view.get("plan") or {}
    candidate = view.get("candidate") or {}
    counts = candidate.get("counts") or {}
    parts = [f"Dataset: {dataset}", f"Status: {run.get('status', 'ready')}"]
    if plan:
        parts.append(f"Tasks: {plan.get('leaf_count', 0)} · {plan.get('phase')}")
    if counts:
        parts.append(f"Accepted: {counts.get('accepted', 0)} · shortfall: {counts.get('shortfall', 0)}")
    if view.get("delivery_path"):
        parts.append(f"Delivery: {view['delivery_path']}")
    return " | ".join(parts)


def event_view(value):
    try:
        event = json.loads(value) if isinstance(value, (str, bytes)) else value
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(event, dict):
        return None
    agent = event.get("agent")
    kind = event.get("event_type")
    if not isinstance(agent, str) or kind not in {
        "agent_step_started", "tool_started", "tool_result", "llm_call", "llm_validation_error",
        "coding_worker_event", "agent_started", "agent_finished", "worker_started", "worker_result",
        "publish_started", "publish_finished", "candidate_submitted",
        "stage_started", "stage_validated", "stage_completed", "stage_paused"
    }:
        return None
    operation = event.get("operation", "")
    status = event.get("status", "")
    label = agent.removesuffix("Agent")
    timestamp = event.get("timestamp", "")
    time = timestamp[11:19] if isinstance(timestamp, str) else ""
    dataset = str(event.get("dataset_id") or "")
    source = str(event.get("source_id") or "")
    stage = str(event.get("stage") or event.get("phase") or "")
    attempt = event.get("attempt")
    milestone = str(event.get("milestone") or "")
    progress_count = event.get("processed") or event.get("processed_count")
    if kind == "coding_worker_event":
        operation = str(event.get("operation") or "event")
        detail = f"OpenHands · {operation.removesuffix('Event')}"
        status = event.get("status", "running")
        suffix = f" · {stage}" if stage else ""
        if attempt is not None:
            suffix += f" attempt {attempt}"
        if milestone:
            suffix += f" · {milestone}"
        if progress_count is not None:
            suffix += f" · processed {progress_count}"
        detail += suffix
    elif kind == "agent_step_started":
        detail = "deciding"
    elif kind == "tool_started":
        detail = f"{operation} · running"
    elif kind == "tool_result":
        detail = f"{operation} · {status}: {str(event.get('summary', ''))[:240]}"
    elif kind == "llm_validation_error":
        detail = f"{operation} · invalid model output; repairing"
        status = "failed"
    elif kind in {"agent_started", "worker_started", "publish_started"}:
        detail = f"{operation} · started"
        status = status or "running"
    elif kind in {"agent_finished", "worker_result", "publish_finished", "candidate_submitted"}:
        detail = f"{operation} · {status}: {str(event.get('summary') or event.get('diagnostic') or '')[:240]}"
    elif kind in {"stage_started", "stage_validated", "stage_completed", "stage_paused"}:
        stage_label = f"{stage} " if stage else ""
        detail = f"{stage_label}{kind.removeprefix('stage_')} · {status}"
        if attempt is not None:
            detail += f" · attempt {attempt}"
    else:
        latency = event.get("latency_ms")
        duration = f" ({latency / 1000:.1f}s)" if isinstance(latency, (int, float)) else ""
        detail = f"{operation} · model {status}{duration}"
    reason = str(event.get("reason") or "")
    if reason and agent == "SupervisorAgent" and kind == "tool_result":
        detail += f" · {reason}"
    return {"agent": agent, "kind": kind, "dataset": dataset, "source": source,
            "operation": operation, "status": status, "detail": detail,
            "text": f"{time}  {label}: {detail}", "label": label,
            "input_tokens": event.get("input_tokens"), "model": event.get("model"),
            "context_window_tokens": event.get("context_window_tokens")}
