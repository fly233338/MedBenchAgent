"""UI-facing actions for one Run over the shared Compilation runner."""
from pathlib import Path

from medbenchagent.runtime import sessions
from medbenchagent.runtime.graph_runner import execute_compilation, inspect_compilation


class ConversationService:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def record(self):
        return sessions.load_session(self.directory)

    def submit_draft(self, draft):
        sessions.freeze_session(self.directory, draft)
        return self.record()

    def save_draft(self, draft):
        sessions.save_draft(self.directory, draft)

    def run(self, *, resume=False, answer=None, question_id=None, stop_requested=None):
        for event in execute_compilation(self.directory, resume=resume, answer=answer,
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
                sessions.add_message(self.directory, message)
            yield event

    def inspect(self):
        if not (self.directory / "run.json").exists():
            return {"status": "draft", "run_directory": str(self.directory), "enabled_agents": [], "artifacts": [], "history": [], "questions": []}
        return inspect_compilation(self.directory)

    def activity_reader(self):
        from medbenchagent.runtime.activity import ActivityReader
        return ActivityReader(sessions.session_directory(self.directory))
