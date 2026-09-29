"""Public service for standalone and embedded preparation runs."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from medbenchagent.runtime.run_artifacts import atomic_write_json
from .preparation_controller import PreparationController


class PreparationService:
    def __init__(self, *, runtime=None):
        self.controller = PreparationController(runtime=runtime)

    def create_or_resume(self, request: dict[str, Any]) -> dict[str, Any]:
        workspace = Path(request["workspace"]).resolve()
        state_path = workspace / "state" / "session.json"
        if state_path.is_file():
            return json.loads(state_path.read_text(encoding="utf-8"))
        session = {"schema_version": "preparation-session.v3", "dataset_id": request["dataset_id"],
                   "revision": request.get("revision", "1"), "conversation_id": str(uuid.uuid4()),
                   "workspace": str(workspace), "sources": request.get("sources", []),
                   "goal": request.get("goal", ""), "stage": "inventory", "status": "running",
                   "turn_counter": 0, "artifacts": {}, "questions": [], "decisions": []}
        atomic_write_json(state_path, session)
        return session

    def inspect(self, ref):
        return self._load(ref)

    def step(self, ref, *, stop_requested=None):
        session = self._load(ref)
        if session["status"] in {"complete", "cancelled", "blocked"}:
            return session
        result = self.controller.step(session, stop_requested=stop_requested)
        updated = self.controller.advance(session, result)
        atomic_write_json(Path(session["workspace"]) / "state" / "session.json", updated)
        return updated

    def run_until_boundary(self, ref, *, stop_requested=None):
        current = self._load(ref)
        while current["status"] == "running":
            current = self.step(current, stop_requested=stop_requested)
        return current

    def answer(self, ref, answer):
        session = self._load(ref)
        session["decisions"].append(answer if isinstance(answer, dict) else {"answer_text": str(answer)})
        session["status"] = "running"
        atomic_write_json(Path(session["workspace"]) / "state" / "session.json", session)
        return session

    def cancel(self, ref):
        session = self._load(ref)
        session["status"] = "cancelled"
        atomic_write_json(Path(session["workspace"]) / "state" / "cancel_requested.json", {"cancelled": True})
        atomic_write_json(Path(session["workspace"]) / "state" / "session.json", session)
        return session

    @staticmethod
    def _load(ref):
        path = Path(ref["workspace"] if isinstance(ref, dict) else ref) / "state" / "session.json"
        return json.loads(path.read_text(encoding="utf-8"))
