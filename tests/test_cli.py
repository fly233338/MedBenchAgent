from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.types import Interrupt

from medbenchagent.cli import _consume, _parser
from medbenchagent.runtime.graph_runner import GraphRunner, RunEvent


def test_parser_exposes_terminal_commands() -> None:
    parser = _parser()
    assert parser.parse_args([]).command is None
    assert parser.parse_args(["create", "pilot"]).experiment == "pilot"
    assert parser.parse_args(["run", "pilot", "build a benchmark"]).command == "run"
    assert parser.parse_args(["resume", "pilot", "run-id"]).run_id == "run-id"
    with pytest.raises(SystemExit):
        parser.parse_args(["resume-intake", "pilot"])


def test_runner_emits_initialized_and_clarification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("medbenchagent.runtime.preflight.preflight_errors", lambda *_args, **_kw: [])
    monkeypatch.setattr("medbenchagent.runtime.preflight.rag_capability",
                        lambda _settings: {"available": True})
    question = Interrupt(value={"question": "Which language?"}, id="question-1")

    class FakeGraph:
        def stream(self, *_args, **_kwargs):
            yield (), "updates", {"initialize": {
                "run_id": "graph-run", "run_directory": str(tmp_path / "graph-run"),
                "effective_config_path": str(tmp_path / "graph-run" / "effective_config.json"),
                "construction_events_path": str(tmp_path / "graph-run" / "events.jsonl"),
            }}
            yield (), "updates", {"__interrupt__": (question,)}

        def get_state(self, _config):
            return SimpleNamespace(values={"run_directory": str(tmp_path / "graph-run"),
                                           "context": {}}, next=("wait",), tasks=())

    runner = GraphRunner(graph_factory=lambda **_kwargs: FakeGraph())
    settings = SimpleNamespace(paths=SimpleNamespace(runs_root=tmp_path))
    run = {"id": "session-run", "thread_id": "thread-1", "requirement": "build"}
    events = list(runner.execute(run, "pilot", tmp_path, settings=settings))
    assert [event.kind for event in events if event.kind in {"initialized", "needs_input"}] == [
        "initialized", "needs_input"]
    assert events[-1].payload["question"] == "Which language?"


@pytest.mark.parametrize("stop_at_start", [False, True])
def test_runner_drains_and_resumes_without_repeating_work(tmp_path, monkeypatch, stop_at_start):
    from langgraph.graph import END, START, StateGraph
    from medbenchagent.settings import active_settings

    monkeypatch.setattr("medbenchagent.runtime.preflight.preflight_errors", lambda *a, **kw: [])
    monkeypatch.setattr("medbenchagent.runtime.preflight.rag_capability", lambda s: {"available": True})
    monkeypatch.setattr("medbenchagent.runtime.dependencies.verify_resume", lambda c: None)
    settings = SimpleNamespace(paths=SimpleNamespace(runs_root=tmp_path), harness={"max_calls": 500})
    monkeypatch.setattr("medbenchagent.runtime.graph_runner.run_settings", lambda path: active_settings(settings))
    calls = []

    def factory(checkpointer):
        graph = StateGraph(dict)

        def initialize(state):
            return {"run_directory": str(tmp_path / "run"), "context": {
                "effective_config_path": "frozen.json", "limits": settings.harness}}

        def work(state):
            calls.append("work")
            if not stop_at_start:
                runner.request_stop()
            return state

        graph.add_node("initialize", initialize)
        graph.add_node("work", work)
        graph.add_node("finish", lambda state: {**state, "status": "complete"})
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "work")
        graph.add_edge("work", "finish")
        graph.add_edge("finish", END)
        return graph.compile(checkpointer=checkpointer)

    runner = GraphRunner(graph_factory=factory)
    run = {"id": "run", "thread_id": "drain", "requirement": "build", "runs_root": str(tmp_path)}
    events = list(runner.execute(run, "pilot", tmp_path, settings=settings,
                                 stop_requested=lambda: stop_at_start))
    assert events[-1].payload["status"] == "interrupted"
    if stop_at_start:
        assert not calls
    resumed = list(runner.execute(run, "pilot", tmp_path, resume=True))
    assert resumed[-1].payload["status"] == "complete"
    assert calls == ["work"]


def test_cli_many_clarifications_use_iteration(monkeypatch):
    count = 0

    class Service:
        def answer(self, run_id, text):
            pass

        def resume(self, run_id, answer=None):
            nonlocal count
            count += 1
            if count < 1100:
                yield RunEvent("r", "needs_input", {"question": "Clarify"})
            else:
                yield RunEvent("r", "finished", {"status": "complete"})

    monkeypatch.setattr("builtins.input", lambda prompt: "answer")
    assert _consume(Service(), iter([RunEvent("r", "needs_input", {"question": "Clarify"})])) == 0
    assert count == 1100
