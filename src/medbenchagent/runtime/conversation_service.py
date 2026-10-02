"""Persistent conversations and serial independent workflow runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from filelock import FileLock, Timeout

from medbenchagent.runtime import sessions
from medbenchagent.runtime.graph_runner import GraphRunner, RunEvent
from medbenchagent.settings import load_settings


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
        self.submit_lock = FileLock(str(self.folder / "submit.lock"))
        self.run_lock = FileLock(str(self.folder / "run.lock"))

    def record(self) -> dict:
        with self.session_lock:
            return sessions.load_experiment(self.name, root=self.root)

    def submit(self, text: str) -> Submission:
        if not text.strip():
            raise ValueError("Enter a request")
        with self.submit_lock, self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            run = sessions.queue_run(record, text.strip(), root=self.root, save=False)
            sessions.add_message(record, "user", text.strip(), run_id=run["id"], root=self.root, save=False)
            sessions.save_experiment(record, root=self.root)
            return Submission("", (run["id"],))

    def answer(self, run_id: str, text: str) -> None:
        if not text.strip():
            raise ValueError("Clarification answer cannot be empty")
        with self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            run = next((item for item in record["runs"] if item["id"] == run_id), None)
            if run is None or run["status"] != "waiting_input":
                raise ValueError("Run is not waiting for an answer")
            if run.get("pending_answer"):
                if run["pending_answer"] != text:
                    raise ValueError("This run question already has an answer")
                return
            sessions.add_message(record, "user", text, run_id=run_id, root=self.root, save=False)
            run["pending_answer"] = text
            sessions.save_experiment(record, root=self.root)

    def reconcile(self) -> dict:
        """Repair stale running entries after a process exit."""
        try:
            with self.run_lock.acquire(timeout=0):
                with self.session_lock:
                    record = sessions.load_experiment(self.name, root=self.root)
                    stale_runs = [item for item in record["runs"] if item["status"] == "running"]
                    if not stale_runs:
                        return record
                    inspected = self.runner.inspect_many(stale_runs)
                    for run in stale_runs:
                        changes = inspected[run["id"]]
                        if run.get("summary") and changes.get("status") == "interrupted":
                            changes.pop("summary", None)
                        updated = sessions.update_run(record, run["id"], root=self.root, **changes)
                        if updated["status"] in {"complete", "blocked", "failed"}:
                            reply = f"Run {updated['status']}. {updated.get('summary', '')}".strip()
                            sessions.add_message(record, "assistant", reply, run_id=run["id"], root=self.root)
                    return record
        except Timeout:
            return self.record()

    def drive(self, *, stop_requested: Callable[[], bool] | None = None) -> Iterator[RunEvent]:
        """Execute independent queued runs until a clarification is needed or the queue empties."""
        try:
            with self.run_lock.acquire(timeout=0):
                while True:
                    if stop_requested is not None and stop_requested():
                        return
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
                    for event in self.runner.execute(run, self.name, self.root, settings=settings,
                                                     stop_requested=stop_requested):
                        self._apply(event)
                        yield event
                        if event.kind == "needs_input" or (
                                event.kind == "error" and event.payload.get("status") == "interrupted"):
                            stopped = True
                    if stopped:
                        return
        except Timeout:
            return

    def resume(self, run_id: str, *, answer: str | None = None,
               stop_requested: Callable[[], bool] | None = None) -> Iterator[RunEvent]:
        if answer is not None:
            self.answer(run_id, answer)
        with self.run_lock.acquire(timeout=0):
            with self.session_lock:
                record = sessions.load_experiment(self.name, root=self.root)
                run = next((item for item in record["runs"] if item["id"] == run_id), None)
                if run is None:
                    raise ValueError(f"Unknown run: {run_id}")
                if run["status"] not in {"interrupted", "waiting_input", "running"}:
                    raise ValueError(f"Run cannot be resumed from {run['status']}")
                sessions.update_run(record, run_id, root=self.root, status="running")
            for event in self.runner.execute(run, self.name, self.root, resume=True, answer=answer,
                                             stop_requested=stop_requested):
                self._apply(event)
                yield event
        yield from self.drive(stop_requested=stop_requested)

    def _apply(self, event: RunEvent) -> None:
        if event.kind not in {"initialized", "needs_input", "finished", "error"}:
            return
        with self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            run = next(item for item in record["runs"] if item["id"] == event.run_session_id)
            changes = {key: value for key, value in event.payload.items() if value is not None
                       if key in {"run_id", "run_directory", "runs_root", "construction_events_path",
                                  "effective_config_path", "catalog_snapshot_path", "selected_assets",
                                   "bir_path", "bir_revision", "bir_manifest_digest",
                                   "bir_template_revision", "bir_summary_ref", "summary", "status"}}
            if event.kind == "needs_input":
                changes["status"] = "waiting_input"
            elif event.kind == "error":
                changes["status"] = event.payload.get("status", "failed")
            run.update(changes)
            for key in ("bir_path", "bir_revision", "bir_manifest_digest", "bir_template_revision"):
                if key in changes and changes[key] is not None:
                    record[key] = changes[key]
            if event.kind == "needs_input":
                question = event.payload["question"]
                if run.get("pending_question") != question or run.get("pending_answer"):
                    sessions.add_message(record, "assistant", question, run_id=run["id"], root=self.root, save=False)
                run["pending_question"] = question
                run["pending_answer"] = None
            if event.kind in {"finished", "error"}:
                run["pending_question"] = None
                run["pending_answer"] = None
                reply = f"Run {run['status']}. {run.get('summary', '')}".strip()
                sessions.add_message(record, "assistant", reply, run_id=run["id"], root=self.root, save=False)
            sessions.save_experiment(record, root=self.root)
