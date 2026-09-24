"""Persistent conversation and serial run orchestration shared by CLI and TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout

from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.runtime import sessions
from medbenchagent.runtime.graph_runner import GraphRunner, RunEvent
from medbenchagent.runtime.session_supervisor import classify_requirement
from medbenchagent.settings import active_settings, load_settings


@dataclass(frozen=True)
class Submission:
    reply: str
    run_ids: tuple[str, ...]


class ConversationService:
    def __init__(self, name: str, *, root: Path = sessions.EXPERIMENTS_ROOT,
                 runner: GraphRunner | None = None) -> None:
        self.name = name
        self.root = root
        self.runner = runner or GraphRunner()
        self.folder = sessions.experiment_path(name, root)
        self.session_lock = FileLock(str(self.folder / "session.json.lock"))
        self.run_lock = FileLock(str(self.folder / "run.lock"))

    def record(self) -> dict:
        with self.session_lock:
            return sessions.load_experiment(self.name, root=self.root)

    def submit(self, text: str) -> Submission:
        if not text.strip():
            raise ValueError("Enter a request")
        with self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            history = list(record["messages"])
            sessions.add_message(record, "user", text, root=self.root)
        settings = load_settings(sessions.config_path(self.name, root=self.root))
        with active_settings(settings):
            known = [{"dataset_id": item["dataset_id"], "name": item["name"]}
                     for item in load_dataset_catalog()]
            known.extend({"dataset_id": item["dataset_id"], "name": item["name"]}
                         for item in settings.dataset_sources)
            decision = classify_requirement(text, known_datasets=known, history=history)
        with self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            if decision.reply:
                sessions.add_message(record, "assistant", decision.reply, root=self.root)
            dependency = None
            run_ids = []
            for task in decision.tasks:
                run = sessions.queue_run(record, task.kind, text, task.dataset_id,
                                         task.requested_dataset_ids, dependency, root=self.root)
                dependency = run["id"]
                run_ids.append(run["id"])
        return Submission(decision.reply, tuple(run_ids))

    def answer(self, run_id: str, text: str) -> None:
        if not text.strip():
            raise ValueError("Clarification answer cannot be empty")
        with self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            run = next((item for item in record["runs"] if item["id"] == run_id), None)
            if run is None or run["status"] != "waiting_input":
                raise ValueError("Run is not waiting for an answer")
            sessions.add_message(record, "user", text, run_id=run_id, root=self.root)

    def reconcile(self) -> dict:
        """Repair stale running entries after a process exit; an active owner keeps its lock."""
        try:
            with self.run_lock.acquire(timeout=0):
                with self.session_lock:
                    record = sessions.load_experiment(self.name, root=self.root)
                    for run in record["runs"]:
                        if run["status"] != "running":
                            continue
                        try:
                            changes = self.runner.inspect(run)
                        except Exception as exc:
                            changes = {"status": "interrupted",
                                       "summary": f"Could not inspect checkpoint: {exc}"}
                        if run.get("summary") and changes.get("status") == "interrupted":
                            changes.pop("summary", None)
                        sessions.update_run(record, run["id"], root=self.root, **changes)
                    return record
        except Timeout:
            return self.record()

    def drive(self) -> Iterator[RunEvent]:
        """Execute independent queued runs until a clarification is needed or the queue empties."""
        try:
            with self.run_lock.acquire(timeout=0):
                while True:
                    with self.session_lock:
                        record = sessions.load_experiment(self.name, root=self.root)
                        if any(run["status"] in {"running", "interrupted", "waiting_input"}
                               for run in record["runs"]):
                            return
                        run = sessions.pop_pending(record, root=self.root)
                    if run is None:
                        return
                    try:
                        settings = load_settings(sessions.config_path(self.name, root=self.root))
                    except Exception as exc:
                        event = RunEvent(run["id"], "error", {
                            "status": "blocked", "summary": f"Invalid experiment configuration: {exc}",
                        })
                        self._apply(event)
                        yield event
                        continue
                    with self.session_lock:
                        record = sessions.load_experiment(self.name, root=self.root)
                        sessions.update_run(record, run["id"], root=self.root, status="running",
                                            runs_root=str(settings.paths.runs_root))
                    stopped = False
                    for event in self.runner.execute(run, self.name, self.root, settings=settings):
                        self._apply(event)
                        yield event
                        if event.kind == "needs_input" or (
                            event.kind == "error" and event.payload.get("status") == "interrupted"
                        ):
                            stopped = True
                    if stopped:
                        return
        except Timeout:
            return

    def resume(self, run_id: str, *, answer: str | None = None) -> Iterator[RunEvent]:
        with self.run_lock.acquire(timeout=0):
            with self.session_lock:
                record = sessions.load_experiment(self.name, root=self.root)
                run = next((item for item in record["runs"] if item["id"] == run_id), None)
                if run is None:
                    raise ValueError(f"Unknown run: {run_id}")
                if run["status"] not in {"interrupted", "waiting_input", "running"}:
                    raise ValueError(f"Run cannot be resumed from {run['status']}")
                sessions.update_run(record, run_id, root=self.root, status="running")
            for event in self.runner.execute(run, self.name, self.root, resume=True, answer=answer):
                self._apply(event)
                yield event

    def _apply(self, event: RunEvent) -> None:
        if event.kind not in {"initialized", "needs_input", "finished", "error"}:
            return
        with self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            run = next(item for item in record["runs"] if item["id"] == event.run_session_id)
            changes = {key: value for key, value in event.payload.items() if value is not None
                       if key in {"run_id", "run_directory", "runs_root", "effective_config_path",
                                  "catalog_snapshot_path", "source_decision_path", "selected_assets",
                                  "summary", "status"}}
            if event.kind == "needs_input":
                changes["status"] = "waiting_input"
            elif event.kind == "error":
                changes["status"] = event.payload.get("status", "failed")
            sessions.update_run(record, event.run_session_id, root=self.root, **changes)
            if event.kind in {"finished", "error"}:
                current = next(item for item in record["runs"] if item["id"] == run["id"])
                reply = f"{current['kind']}: {current['status']}. {current.get('summary', '')}"
                sessions.add_message(record, "assistant", reply, run_id=run["id"], root=self.root)
                if current["status"] in {"blocked", "failed"}:
                    sessions.block_dependents(record, run["id"], root=self.root)
