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

    def run(self, run_id, *, resume=False, answer=None, question_id=None, stop_requested=None):
        run = next(row for row in self.record()["runs"] if row["id"] == run_id)
        with FileLock(self.folder / "execution.lock", timeout=0):
            for event in execute_compilation(run["directory"], resume=resume, answer=answer,
                                             question_id=question_id, stop_requested=stop_requested):
                message = None
                if event.kind == "needs_input":
                    question = event.payload["questions"][0]
                    message = {"id": "question-" + event.payload["question_id"], "role": "assistant",
                               "text": question.get("question", str(question)), "kind": "question",
                               "after_work": len(event.payload["history"])}
                elif event.kind == "answer_accepted":
                    message = {"id": "answer-" + event.payload["question_id"], "role": "user",
                               "text": event.payload["answer"], "after_work": event.payload["after_work"]}
                if message:
                    with FileLock(self.folder / "session.lock"):
                        record = self.record()
                        if not any(row.get("id") == message["id"] and row.get("run_id") == run_id for row in record["messages"]):
                            record["messages"].append({**message, "run_id": run_id})
                            sessions.save_experiment(record, root=self.root)
                yield event

    def inspect(self, run_id):
        run = next(row for row in self.record()["runs"] if row["id"] == run_id)
        return inspect_compilation(run["directory"])

    def activity_reader(self, run_id):
        from medbenchagent.runtime.activity import ActivityReader
        run = next(row for row in self.record()["runs"] if row["id"] == run_id)
        return ActivityReader(run["directory"])
