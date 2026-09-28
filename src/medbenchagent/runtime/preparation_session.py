"""Persistent state and bounded context for long lived preparation sessions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SESSION_SCHEMA = "preparation-session.v2"
SUMMARY_SCHEMA = "preparation-context.v1"


def digest(value: str | bytes | Path) -> str:
    if isinstance(value, Path):
        data = value.read_bytes()
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = value
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def session_paths(workspace: Path) -> dict[str, Path]:
    state = workspace / "state"
    return {"state": state / "session_state.json", "summary": state / "context_summary.json",
            "task": state / "active_task.json", "feedback": state / "last_feedback.json"}


def load_session(workspace: Path) -> dict[str, Any]:
    path = session_paths(workspace)["state"]
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_session(workspace: Path, *, task: dict[str, Any], status: str,
                  step: str = "", attempt: int = 1, source_digest: str = "",
                  facts: list[str] | None = None, unresolved: list[str] | None = None,
                  feedback: dict[str, Any] | None = None, stop_reason: str = "") -> dict[str, Any]:
    previous = load_session(workspace)
    state = {"schema_version": SESSION_SCHEMA, "task_id": task.get("task_id", ""),
             "kind": task.get("kind", "preparation"), "source_root": task.get("source_root", ""),
             "source_digest": source_digest or previous.get("source_digest", ""),
             "workflow_step": step or previous.get("workflow_step", ""), "attempt": attempt,
             "conversation_id": task.get("conversation_id") or previous.get("conversation_id", ""),
             "status": status, "facts": facts if facts is not None else previous.get("facts", []),
             "unresolved": unresolved if unresolved is not None else previous.get("unresolved", []),
             "last_feedback": feedback or previous.get("last_feedback", {}),
             "observed_files": previous.get("observed_files", []),
             "budget": {"max_iterations": task.get("max_iterations"),
                        "timeout_seconds": task.get("timeout_seconds")},
             "stop_reason": stop_reason}
    _write(session_paths(workspace)["state"], state)
    return state


def write_summary(workspace: Path, *, objective: str, facts: list[str], inferred: list[str] | None = None,
                  unresolved: list[str] | None = None, user_decisions: list[str] | None = None,
                  current_task: str = "", validator_feedback: dict[str, Any] | None = None,
                  artifacts: list[str] | None = None, max_chars: int = 8000) -> dict[str, Any]:
    summary = {"schema_version": SUMMARY_SCHEMA, "objective": objective[:2000],
               "verified_facts": facts[:100], "inferences": (inferred or [])[:50],
               "unresolved": (unresolved or [])[:50], "user_decisions": (user_decisions or [])[:50],
               "current_task": current_task[:2000], "validator_feedback": validator_feedback or {},
               "artifacts": (artifacts or [])[:100]}
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    if len(encoded) > max_chars:
        summary["verified_facts"] = summary["verified_facts"][:30]
        summary["artifacts"] = summary["artifacts"][:30]
        summary["compacted"] = True
    _write(session_paths(workspace)["summary"], summary)
    return summary


def bounded_feedback(value: str, limit: int = 6000) -> str:
    value = value or ""
    return value if len(value) <= limit else value[:limit] + "\n[truncated; full report is in workspace]"

