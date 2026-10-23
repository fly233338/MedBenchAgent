"""UI-facing experiment actions over the shared Compilation runner."""
from pathlib import Path
from filelock import FileLock
from medbenchagent.runtime import sessions
from medbenchagent.runtime.graph_runner import execute_compilation, inspect_compilation


class ConversationService:
    def __init__(self, name, *, root=sessions.EXPERIMENTS_ROOT):
        self.name, self.root = name, Path(root)
        self.folder = sessions.experiment_path(name, root)

    def record(self):
        return sessions.load_experiment(self.name, root=self.root)

    def submit(self, goal, **options):
        return sessions.add_run(self.name, goal, root=self.root, **options)

    def submit_draft(self, draft):
        return sessions.add_run(self.name, draft.goal, root=self.root, draft=draft)

    def save_draft(self, draft):
        sessions.save_draft(self.name, draft, root=self.root)

    def run(self, run_id, *, resume=False, answer=None, stop_requested=None):
        run = next(row for row in self.record()["runs"] if row["id"] == run_id)
        for event in execute_compilation(run["directory"], resume=resume, answer=answer, stop_requested=stop_requested):
            if event.kind in {"finished", "needs_input"}:
                with FileLock(self.folder / "session.lock"):
                    record = self.record()
                    text = event.payload.get("summary", "") or str(event.payload.get("questions", []))
                    record["messages"].append({"role": "assistant", "text": text, "run_id": run_id})
                    sessions.save_experiment(record, root=self.root)
            yield event

    def inspect(self, run_id):
        run = next(row for row in self.record()["runs"] if row["id"] == run_id)
        return inspect_compilation(run["directory"])

    def activity_reader(self, run_id):
        from medbenchagent.runtime.activity import ActivityReader
        run = next(row for row in self.record()["runs"] if row["id"] == run_id)
        return ActivityReader(run["directory"])
