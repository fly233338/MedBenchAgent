from runtime_fixtures import runtime_for
from contextlib import contextmanager
from pathlib import Path

import pytest
from medbenchagent.compilation.contracts import Delegate, Finish, AskUser, WorkRequest, WorkResult
from medbenchagent.compilation.interfaces import AgentSpec, CompilationContext
from medbenchagent.runtime.intake import InputDraft
from medbenchagent.runtime.sessions import create_session
from medbenchagent.runtime.conversation_service import ConversationService


def test_pause_question_resume_and_stale_answer_use_checkpoint(tmp_path, monkeypatch):
    import medbenchagent.runtime.model_services as services
    import medbenchagent.runtime.graph_runner as runner
    monkeypatch.setattr(services, "resolve_model_service", lambda *_: "offline")
    calls = []
    def node(state):
        request = WorkRequest.model_validate(state["request"])
        calls.append(request.request_id)
        return {"result": WorkResult(request_id=request.request_id, outcome="completed", summary="Measured source").model_dump(mode="json")}
    def decide(view):
        if not view.history:
            return Delegate(agent_id="data_processing", operation="profile", objective="Inspect")
        if not view.answers:
            return AskUser(question="Are these explicitly paired?")
        return Finish(goal_result="succeeded", summary="Completed after clarification")
    @contextmanager
    def assembly(options, directory):
        yield {"data_processing": AgentSpec("Source inspection", node, operations=("profile",))}, CompilationContext(directory, decide, options.goal, runtime_directory=Path(options.runtime_directory))
    monkeypatch.setattr(runner, "assemble_compilation", assembly)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notes.txt").write_text("source", encoding="utf-8")
    root = tmp_path / "experiments"
    create_session("default", root=root, runtime_root=runtime_for(root))
    service = ConversationService(root / "default")
    row = service.submit_draft(InputDraft(goal="Inspect", source_dirs=[str(raw)], goal_kind="profile"))
    identifier = row["id"]
    assert service.inspect()["status"] == "ready"
    assert list(service.run( stop_requested=lambda: True))[-1].kind == "paused"
    assert service.inspect()["status"] == "paused" and calls == []
    assert list(service.run( resume=True, stop_requested=lambda: True))[-1].kind == "paused"
    assert len(calls) == 1
    from langgraph.checkpoint.sqlite import SqliteSaver
    from filelock import FileLock
    from medbenchagent.compilation.graph import build_compilation_graph
    with SqliteSaver.from_conn_string(str(Path(row["runtime_directory"]) / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph({"data_processing": AgentSpec("Source inspection", node, operations=("profile",))}, checkpointer=saver)
        graph.update_state({"configurable": {"thread_id": identifier}}, {"status": "running"})
    assert service.inspect()["status"] == "interrupted"
    with FileLock(Path(row["runtime_directory"]) / "locks/execution.lock"):
        assert service.inspect()["status"] == "running"
    events = list(service.run( resume=True))
    waiting = service.inspect()
    assert events[-1].kind == "needs_input" and waiting["status"] == "waiting"
    assert waiting["questions"][0]["question"] == "Are these explicitly paired?"
    with pytest.raises(ValueError, match="different question"):
        list(service.run( resume=True, answer="yes", question_id="stale"))
    assert service.inspect()["question_id"] == waiting["question_id"]
    events = list(service.run( resume=True, answer="yes", question_id=waiting["question_id"]))
    assert any(event.kind == "answer_accepted" for event in events)
    assert service.inspect()["status"] == "stopped" and len(calls) == 1
    assert len([m for m in service.record()["messages"] if m.get("id", "").startswith("answer-")]) == 1
    with pytest.raises(ValueError, match="ended"):
        list(service.run( resume=True))
    assert (Path(row["runtime_directory"]) / "checkpoints.sqlite").exists()
    (Path(row["runtime_directory"]) / "studio-thread.txt").write_text("studio-owner", encoding="utf-8")
    assert service.inspect()["status"] == "external"
    with pytest.raises(ValueError, match="Studio"):
        list(service.run( resume=True))
