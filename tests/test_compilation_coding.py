"""Exercise the actual adapter/turn replay with a tiny durable SDK substitute."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.sqlite import SqliteSaver
import pytest

from medbenchagent.compilation.artifacts import resolve_artifact
from medbenchagent.compilation.coding import OpenHandsCodingAgent
from medbenchagent.compilation.coding_worker import CodingCandidate, execute_turn
from medbenchagent.compilation.contracts import ArtifactRef, Delegate, Finish, WorkRequest, WorkResult
from medbenchagent.compilation.graph import CompilationContext, build_compilation_graph
from medbenchagent.settings import load_coding_settings


class Crash(BaseException):
    """An interrupted process, not a reported professional failure."""


def _event(value):
    if value["source"] == "user":
        return SimpleNamespace(source="user", sender=value["sender"], llm_message=SimpleNamespace(
            role="user", content=[SimpleNamespace(text=value["text"])],
        ))
    return SimpleNamespace(**value)


class FakeFinish:
    def parse_last_response(self, events):
        return next(({"result": item.candidate} for item in reversed(events)
                     if hasattr(item, "candidate")), None)


class FakeConversation:
    def __init__(self, payload, perform, crash):
        self.payload, self.perform, self.crash = payload, perform, crash
        self.path = Path(payload["persistence_dir"]) / "fake.json"
        self.saved = (json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists()
                      else {"events": [], "runs": 0})
        self.state = SimpleNamespace(events=[_event(item) for item in self.saved["events"]],
                                     execution_status="idle")

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
        return CodingCandidate(request_id=payload["request"]["request_id"], outcome="completed",
                               summary="Wrote a file", files=[{"path": "answer.txt", "kind": "report"}])

    def __call__(self, payload):
        self.payloads.append(payload)
        conversation = FakeConversation(payload, self.perform, self.crash)
        return execute_turn(conversation, FakeFinish(), WorkRequest.model_validate(payload["request"]),
                            payload["prompt"])


@pytest.fixture
def coding_settings(monkeypatch):
    import medbenchagent.settings as settings
    monkeypatch.setattr(settings, "load_project_env", lambda: None)
    for name in (*settings._ENVIRONMENT_OVERRIDES, "MEDBENCH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return replace(load_coding_settings(), llm=replace(load_coding_settings().llm, model="test-model"))


def test_workstreams_reuse_isolate_and_exchange_formal_artifacts(tmp_path, coding_settings):
    runner = FakeRunner()
    agent = OpenHandsCodingAgent(coding_settings, runner=runner)
    saw_inputs = []

    def perform(payload):
        if payload["request"]["objective"] == "B":
            # A's mutable output can change without changing the handed-off copy.
            (Path(runner.payloads[0]["output_directory"]) / "answer.txt").write_text("mutated", encoding="utf-8")
        saw_inputs.append([Path(item["read_path"]).read_text(encoding="utf-8")
                           for item in payload["inputs"]])
        return FakeRunner.write_file(payload)

    runner.perform = perform

    def decide(view):
        index = len(view.history)
        if index == 3:
            assert view.workstream_summaries()[0]["last_result"]["request_id"].endswith("000003")
            return Finish(summary="Done")
        return Delegate(
            agent_id="coding", objective=("A", "B", "A again")[index],
            scope=["same" if index < 2 else "refined"], inputs=list(view.last_result.artifacts) if index else [],
            constraints=["Preserve input files"],
            workstream_id=view.workstreams[0].workstream_id if index == 2 else None,
        )

    context = CompilationContext(tmp_path, decide)
    config = {"configurable": {"thread_id": "streams"}}
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph({"coding": agent.spec()}, checkpointer=saver)
        output = graph.invoke({"compilation_id": "streams", "goal": "Share code"}, config,
                              context=context, durability="sync")
    assert [item["outcome"] for item in output["history"]] == ["completed"] * 3
    a, b, again = runner.payloads
    assert a["conversation_id"] == again["conversation_id"] != b["conversation_id"]
    assert a["workspace"] == again["workspace"] != b["workspace"]
    assert a["persistence_dir"] == again["persistence_dir"] != b["persistence_dir"]
    assert saw_inputs == [[], ["A"], ["B"]]
    assert output["workstreams"][0]["scope"] == output["workstreams"][1]["scope"] == ["same"]
    assert "conversation_id" not in json.dumps(output)
    for payload, runs in ((a, 2), (b, 1)):
        saved = json.loads((Path(payload["persistence_dir"]) / "fake.json").read_text())
        assert saved["runs"] == runs
        assert len([event for event in saved["events"] if event["source"] == "user"]) == runs

    request = WorkRequest.model_validate(again["request"])
    ref = ArtifactRef.model_validate(output["artifacts"][-1])
    formal = resolve_artifact(tmp_path, ref)
    original = (formal.read_bytes(), formal.stat().st_mtime_ns)
    result = agent({"request": request.model_dump()}, SimpleNamespace(context=context))
    assert result["result"]["outcome"] == "completed"
    assert (formal.read_bytes(), formal.stat().st_mtime_ns) == original
    (Path(again["output_directory"]) / "answer.txt").write_text("conflict", encoding="utf-8")
    assert agent({"request": request.model_dump()}, SimpleNamespace(context=context))["result"]["outcome"] == "failed"
    assert (formal.read_bytes(), formal.stat().st_mtime_ns) == original

    other = request.model_copy(update={"compilation_id": "another", "request_id": "another-1", "inputs": []})
    other_context = CompilationContext(tmp_path / "another", decide)
    agent({"request": other.model_dump()}, SimpleNamespace(context=other_context))
    assert runner.payloads[-1]["conversation_id"] != a["conversation_id"]
    assert runner.payloads[-1]["workspace"] != a["workspace"]

    # The real Host rejects paths outside this request, even if the file exists.
    external = Path(a["output_directory"]) / "answer.txt"
    malicious = OpenHandsCodingAgent(coding_settings, runner=lambda payload: CodingCandidate(
        request_id=payload["request"]["request_id"], outcome="completed", summary="bad path",
        files=[{"path": str(external), "kind": "report"}],
    ))
    bad_request = request.model_copy(update={"request_id": "bad-path", "inputs": []})
    rejected = malicious({"request": bad_request.model_dump()}, SimpleNamespace(context=context))["result"]
    assert rejected["outcome"] == "failed" and rejected["artifacts"] == []

    formal.unlink()
    missing_request = request.model_copy(update={"request_id": "missing-input", "inputs": [ref]})
    before = len(runner.payloads)
    missing = agent({"request": missing_request.model_dump()}, SimpleNamespace(context=context))["result"]
    assert missing["outcome"] == "needs_input" and missing["needs"][0]["kind"] == "work"
    assert len(runner.payloads) == before

    unavailable = OpenHandsCodingAgent(replace(coding_settings, worker_python=str(tmp_path / "no-python")))
    blocked = unavailable({"request": bad_request.model_dump()}, SimpleNamespace(context=context))["result"]
    assert blocked["outcome"] == "blocked" and blocked["issues"][0]["code"] == "environment_unavailable"


@pytest.mark.parametrize("crash", ["sent", "finished"])
def test_sdk_replay_after_host_send_or_finish_before_graph_checkpoint(tmp_path, coding_settings, crash):
    def decide(view):
        return (Finish(summary="Done") if view.last_result else
                Delegate(agent_id="coding", objective="Produce a report", scope=["free-form"]))

    config = {"configurable": {"thread_id": "recover"}}
    context = CompilationContext(tmp_path, decide)
    runner = FakeRunner(crash=crash)
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph({"coding": OpenHandsCodingAgent(coding_settings, runner=runner).spec()},
                                        checkpointer=saver)
        with pytest.raises(Crash):
            graph.invoke({"compilation_id": "recover", "goal": "Recover a request"}, config,
                         context=context, durability="sync")
        saved = graph.get_state(config).values
        request = saved["request"]
        assert len(saved["workstreams"]) == 1
        assert request["workstream_id"] == saved["workstreams"][0]["workstream_id"]

    # Fresh graph, adapter and fake SDK objects load only on-disk state.
    restored = FakeRunner()
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph({"coding": OpenHandsCodingAgent(coding_settings, runner=restored).spec()},
                                        checkpointer=saver)
        after = graph.invoke(None, config, context=context, durability="sync")
    assert after["status"] == "completed"
    assert after["request"] == request and after["workstreams"] == saved["workstreams"]
    assert WorkResult.model_validate(after["result"]).outcome == "completed"
    payload = restored.payloads[0]
    persisted = json.loads((Path(payload["persistence_dir"]) / "fake.json").read_text())
    assert persisted["runs"] == 1
    assert len([event for event in persisted["events"] if event["source"] == "user"]) == 1
    assert payload["conversation_id"] == runner.payloads[0]["conversation_id"]
