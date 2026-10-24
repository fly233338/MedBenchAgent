import json
from pathlib import Path
from types import SimpleNamespace

from medbenchagent.agents.data_processing.coding.worker import CodingCandidate, execute_turn
from medbenchagent.compilation.contracts import WorkRequest


class Crash(BaseException):
    """An interrupted process, not a reported professional failure."""


def _event(value):
    if value["source"] == "user":
        return SimpleNamespace(
            source="user",
            sender=value["sender"],
            llm_message=SimpleNamespace(
                role="user",
                content=[SimpleNamespace(text=value["text"])],
            ),
        )
    return SimpleNamespace(**value)


class FakeFinish:
    def parse_last_response(self, events):
        return next(
            ({"result": item.candidate} for item in reversed(events) if hasattr(item, "candidate")),
            None,
        )


class FakeConversation:
    def __init__(self, payload, perform, crash):
        self.payload, self.perform, self.crash = payload, perform, crash
        self.path = Path(payload["persistence_dir"]) / "fake.json"
        self.saved = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"events": [], "runs": 0}
        )
        self.state = SimpleNamespace(
            events=[_event(item) for item in self.saved["events"]],
            execution_status="idle",
        )

    def _save(self, event):
        self.saved["events"].append(event)
        self.state.events.append(_event(event))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.saved), encoding="utf-8")

    def send_message(self, message, *, sender):
        self._save({"source": "user", "sender": sender, "text": message})
        if self.crash == "sent":
            raise Crash()

    def run(self):
        self.saved["runs"] += 1
        result = self.perform(self.payload)
        self._save({"source": "agent", "candidate": result.model_dump(mode="json")})
        self.state.execution_status = "finished"
        if self.crash == "finished":
            raise Crash()


class FakeRunner:
    def __init__(self, perform=None, *, crash=None):
        self.payloads = []
        self.perform = perform or self.write_file
        self.crash = crash

    @staticmethod
    def write_file(payload):
        path = Path(payload["output_directory"]) / "answer.txt"
        path.write_text(payload["request"]["objective"], encoding="utf-8")
        return CodingCandidate(
            request_id=payload["request"]["request_id"],
            outcome="completed",
            summary="Wrote a file",
            files=[{"path": "answer.txt", "kind": "report"}],
        )

    def __call__(self, payload):
        self.payloads.append(payload)
        conversation = FakeConversation(payload, self.perform, self.crash)
        return execute_turn(
            conversation,
            FakeFinish(),
            WorkRequest.model_validate(payload["request"]),
            payload["prompt"],
        )
