from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.types import Interrupt

from medbenchagent.cli import _parser
from medbenchagent.runtime.graph_runner import GraphRunner


def test_parser_exposes_terminal_commands() -> None:
    parser = _parser()
    assert parser.parse_args([]).command is None
    assert parser.parse_args(["create", "pilot"]).experiment == "pilot"
    assert parser.parse_args(["run", "pilot", "build a benchmark"]).command == "run"
    assert parser.parse_args(["resume", "pilot", "run-id"]).run_id == "run-id"
    assert parser.parse_args(["register", "--dataset-id", "a", "--bundle", "b"]).command == "register"
    with pytest.raises(SystemExit):
        parser.parse_args(["build"])


def test_runner_emits_initialized_and_clarification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("medbenchagent.runtime.graph_runner.preflight_errors", lambda *_args, **_kw: [])
    monkeypatch.setattr("medbenchagent.runtime.graph_runner.rag_capability",
                        lambda _settings: {"available": True})
    question = Interrupt(value={"question": "Which language?"}, id="question-1")

    class FakeGraph:
        def stream(self, *_args, **_kwargs):
            yield (), "updates", {"initialize": {
                "run_id": "graph-run", "run_directory": str(tmp_path / "graph-run"),
                "effective_config_path": str(tmp_path / "graph-run" / "effective_config.json"),
            }}
            yield (), "updates", {"__interrupt__": (question,)}

        def get_state(self, _config):
            return SimpleNamespace(values={"run_directory": str(tmp_path / "graph-run"),
                                           "context": {}}, next=("wait",), tasks=())

    runner = GraphRunner(graph_factory=lambda **_kwargs: FakeGraph())
    settings = SimpleNamespace(paths=SimpleNamespace(runs_root=tmp_path))
    run = {"id": "session-run", "thread_id": "thread-1", "kind": "build_benchmark",
           "requirement": "build", "dataset_id": None, "requested_dataset_ids": None}
    events = list(runner.execute(run, "pilot", tmp_path, settings=settings))
    assert [event.kind for event in events if event.kind in {"initialized", "needs_input"}] == [
        "initialized", "needs_input"]
    assert events[-1].payload["question"] == "Which language?"
