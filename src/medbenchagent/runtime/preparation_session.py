"""Persistent state and bounded context for long lived preparation sessions."""

from __future__ import annotations

import hashlib
import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SESSION_SCHEMA = "preparation-session.v2"


@dataclass(frozen=True)
class CodingResult:
    """Stable business result returned by a long-lived preparation session."""
    status: str
    conversation_id: str
    workspace: str
    candidate: str = ""
    question: str = ""
    diagnostic: str = ""


class CodingSession:
    """Small resumable facade around the OpenHands task runner.

    ``runner`` receives the user-visible message and the persisted session state.
    Keeping this adapter independent lets local and Docker workers share the same
    lifecycle and makes the contract straightforward to test with a fake runner.
    """

    def __init__(self, workspace: Path, conversation_id: str, runner):
        self.workspace = Path(workspace)
        self.conversation_id = conversation_id
        self.runner = runner
        self._cancelled = False

    async def run(self, task: str) -> CodingResult:
        return await self._execute(task)

    async def continue_with(self, message: str) -> CodingResult:
        return await self._execute(message)

    async def _execute(self, message: str) -> CodingResult:
        if self._cancelled:
            return CodingResult("blocked", self.conversation_id, str(self.workspace),
                                diagnostic="Coding session was cancelled")
        payload = {"conversation_id": self.conversation_id, "message": message}
        write_session(self.workspace, task=payload, status="running", step="coding")
        try:
            result = await asyncio.to_thread(self.runner, payload)
        except Exception as exc:  # runner errors become a diagnosable blocked state
            write_session(self.workspace, task=payload, status="blocked", step="coding",
                          stop_reason=str(exc))
            return CodingResult("blocked", self.conversation_id, str(self.workspace), diagnostic=str(exc))
        status = result.get("status", "blocked") if isinstance(result, dict) else "blocked"
        write_session(self.workspace, task=payload, status=status, step="coding",
                      stop_reason=result.get("diagnostic", "") if isinstance(result, dict) else "")
        return CodingResult(status=status, conversation_id=self.conversation_id,
                            workspace=str(self.workspace), candidate=result.get("candidate", ""),
                            question=result.get("question", ""),
                            diagnostic=result.get("diagnostic", ""))

    async def cancel(self) -> None:
        self._cancelled = True
        cancel = getattr(self.runner, "cancel", None)
        if cancel is not None:
            value = cancel(self.conversation_id)
            if asyncio.iscoroutine(value):
                await value
        state = load_session(self.workspace)
        state.update({"status": "blocked", "stop_reason": "cancelled"})
        _write(session_paths(self.workspace)["state"], state)


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
    return {"state": state / "session_state.json", "task": state / "active_task.json",
            "feedback": state / "last_feedback.json"}


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



