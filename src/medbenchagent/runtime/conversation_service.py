"""Persistent conversation and serial run orchestration shared by CLI and TUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4

from filelock import FileLock, Timeout

from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.runtime import sessions
from medbenchagent.runtime.graph_runner import GraphRunner, RunEvent
from medbenchagent.runtime.preparation_intake import parse_answer, resolve_intake
from medbenchagent.runtime.session_supervisor import SessionDecision, classify_requirement
from medbenchagent.settings import active_settings, load_settings


@dataclass(frozen=True)
class Submission:
    reply: str
    run_ids: tuple[str, ...]
    question: dict | None = None


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

    def pending_intake(self) -> dict | None:
        record = self.record()
        return next((item for item in record["submissions"] if item["status"] == "awaiting_input"), None)

    def _accept(self, record: dict, submission: dict, tasks: list) -> Submission:
        sessions.add_message(record, "user", submission["text"], root=self.root, save=False)
        sessions.add_message(record, "assistant", submission["reply"], root=self.root, save=False)
        for question in submission.get("questions", []):
            if question["answer"] is not None:
                sessions.add_message(record, "assistant", question["prompt"], root=self.root, save=False)
                sessions.add_message(record, "user", question["answer"], root=self.root, save=False)
        dependency = None
        run_ids = []
        for task in tasks:
            run = sessions.queue_run(record, task.kind, submission["text"], task.dataset_id,
                                     task.requested_dataset_ids, dependency, root=self.root, save=False,
                                     dataset_name=task.dataset_name, source_root=task.source_root,
                                     rebuild=task.rebuild)
            dependency = run["id"]
            run_ids.append(run["id"])
        submission["status"] = "accepted"
        if submission.get("questions"):
            submission["question_batch_status"] = "completed"
        submission["run_ids"] = run_ids
        sessions.save_experiment(record, root=self.root)
        return Submission(submission["reply"], tuple(run_ids))

    def submit(self, text: str, *, on_reply_delta: Callable[[str], None] | None = None,
               on_reply_reset: Callable[[], None] | None = None) -> Submission:
        if not text.strip():
            raise ValueError("Enter a request")
        with self.submit_lock:
            with self.session_lock:
                record = sessions.load_experiment(self.name, root=self.root)
                if any(item["status"] == "awaiting_input" for item in record["submissions"]):
                    raise ValueError("Answer or cancel the pending preparation questions first")
                history = list(record["messages"])
                submission = {"id": str(uuid4()), "text": text, "status": "processing",
                              "reply": "", "questions": [], "current": 0, "run_ids": [], "error": ""}
                record["submissions"].append(submission)
                sessions.save_experiment(record, root=self.root)
            try:
                settings = load_settings(sessions.config_path(self.name, root=self.root))
                with active_settings(settings):
                    catalog = load_dataset_catalog()
                    known = [{"dataset_id": item["dataset_id"], "name": item["name"]} for item in catalog]
                    known.extend({"dataset_id": item["dataset_id"], "name": item["name"]}
                                 for item in settings.dataset_sources)
                    callbacks = ({"on_reply_delta": on_reply_delta, "on_reply_reset": on_reply_reset}
                                 if on_reply_delta is not None else {})
                    decision = classify_requirement(text, known_datasets=known, history=history, **callbacks)
                    tasks, questions = resolve_intake(decision, catalog, settings.dataset_sources)
            except Exception as exc:
                with self.session_lock:
                    record = sessions.load_experiment(self.name, root=self.root)
                    failed = next(item for item in record["submissions"] if item["id"] == submission["id"])
                    failed.update(status="failed", error=str(exc))
                    sessions.save_experiment(record, root=self.root)
                raise
            with self.session_lock:
                record = sessions.load_experiment(self.name, root=self.root)
                current = next(item for item in record["submissions"] if item["id"] == submission["id"])
                current["reply"] = decision.reply
                if questions:
                    current.update(status="awaiting_input", decision=decision.model_dump(),
                                   questions=questions, question_batch_status="pending")
                    current["questions"][0]["status"] = "current"
                    sessions.save_experiment(record, root=self.root)
                    return Submission(decision.reply, (), questions[0])
                return self._accept(record, current, tasks)

    def answer_intake(self, text: str) -> Submission:
        if not text.strip():
            raise ValueError("Clarification answer cannot be empty")
        with self.submit_lock:
            with self.session_lock:
                record = sessions.load_experiment(self.name, root=self.root)
                current = next((item for item in record["submissions"] if item["status"] == "awaiting_input"), None)
                if current is None:
                    raise ValueError("No preparation question is pending")
                submission_id = current["id"]
                field = current["questions"][current["current"]]["field"]
            try:
                settings = load_settings(sessions.config_path(self.name, root=self.root))
                with active_settings(settings):
                    value = parse_answer(field, text)
                    if field == "source_root" and not Path(value).expanduser().resolve().is_dir():
                        raise ValueError(f"Raw directory is unavailable: {value}")
                    if field == "dataset_id" and not value.strip():
                        raise ValueError("Dataset ID cannot be empty")
            except Exception as exc:
                with self.session_lock:
                    record = sessions.load_experiment(self.name, root=self.root)
                    current = next(item for item in record["submissions"] if item["id"] == submission_id)
                    current["error"] = str(exc)
                    sessions.save_experiment(record, root=self.root)
                raise
            with self.session_lock:
                record = sessions.load_experiment(self.name, root=self.root)
                current = next(item for item in record["submissions"] if item["id"] == submission_id)
                question = current["questions"][current["current"]]
                question.update(answer=text, status="answered")
                current["question_batch_status"] = "current"
                current["decision"]["tasks"][0][field] = value
                current["current"] += 1
                current["error"] = ""
                if current["current"] < len(current["questions"]):
                    next_question = current["questions"][current["current"]]
                    next_question["status"] = "current"
                    sessions.save_experiment(record, root=self.root)
                    return Submission(current["reply"], (), next_question)
                with active_settings(settings):
                    tasks, questions = resolve_intake(SessionDecision.model_validate(current["decision"]),
                                                      load_dataset_catalog(), settings.dataset_sources)
                if questions:
                    current["questions"].extend(questions)
                    current["questions"][current["current"]]["status"] = "current"
                    sessions.save_experiment(record, root=self.root)
                    return Submission(current["reply"], (), current["questions"][current["current"]])
                return self._accept(record, current, tasks)

    def cancel_intake(self) -> None:
        with self.submit_lock, self.session_lock:
            record = sessions.load_experiment(self.name, root=self.root)
            current = next((item for item in record["submissions"] if item["status"] == "awaiting_input"), None)
            if current is None:
                raise ValueError("No preparation question is pending")
            current["status"] = "cancelled"
            current["question_batch_status"] = "cancelled"
            sessions.save_experiment(record, root=self.root)

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
            with self.submit_lock.acquire(timeout=0):
                with self.session_lock:
                    record = sessions.load_experiment(self.name, root=self.root)
                    stale = [item for item in record["submissions"] if item["status"] == "processing"]
                    for item in stale:
                        item.update(status="failed", error="Submission was interrupted; submit it again")
                    if stale:
                        sessions.save_experiment(record, root=self.root)
        except Timeout:
            pass
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
                        updated = sessions.update_run(record, run["id"], root=self.root, **changes)
                        if updated["status"] in {"complete", "blocked", "failed"}:
                            reply = f"{updated['kind']}: {updated['status']}. {updated.get('summary', '')}"
                            sessions.add_message(record, "assistant", reply, run_id=run["id"],
                                                 root=self.root)
                            if updated["status"] in {"blocked", "failed"}:
                                sessions.block_dependents(record, run["id"], root=self.root)
                    return record
        except Timeout:
            return self.record()

    def drive(self, *, stop_requested: Callable[[], bool] | None = None) -> Iterator[RunEvent]:
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
                    for event in self.runner.execute(run, self.name, self.root, settings=settings,
                                                     stop_requested=stop_requested):
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

    def resume(self, run_id: str, *, answer: str | None = None,
               stop_requested: Callable[[], bool] | None = None) -> Iterator[RunEvent]:
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
