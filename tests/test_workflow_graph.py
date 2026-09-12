from langgraph.checkpoint.memory import MemorySaver
from medbenchagent.harness import compile_agent, result
from medbenchagent.workflow.graph import compile_benchmark_graph


def test_supervisor_replaces_fixed_role_chain():
    graph = compile_benchmark_graph().get_graph()
    assert set(graph.nodes) == {"__start__", "initialize", "supervisor", "__end__"}


def test_harness_selects_tools_and_finishes_without_confirmation(tmp_path, monkeypatch):
    actions = iter(
        [
            {"tool": "compute", "arguments": {}, "reason": "Needed"},
            {"tool": "finish", "arguments": {}, "reason": "Done"},
        ]
    )
    monkeypatch.setattr("medbenchagent.harness.complete_json", lambda *a, **k: next(actions))
    calls = []

    def compute(context):
        calls.append(1)
        return result("success", "Computed", value=3)

    graph = compile_agent(
        "TestAgent",
        "Test",
        {"compute": compute},
        lambda c: c,
        lambda c: result("complete", "Done"),
        checkpointer=MemorySaver(),
    )
    context = {
        "run_id": "run",
        "construction_events_path": str(tmp_path / "events.jsonl"),
        "operation_directory": str(tmp_path / "ops"),
    }
    output = graph.invoke({"context": context}, {"configurable": {"thread_id": "test"}})
    assert output["context"]["value"] == 3
    assert output["result"]["status"] == "complete"
    assert calls == [1]


def test_harness_stops_repeated_invalid_actions(tmp_path, monkeypatch):
    monkeypatch.setattr("medbenchagent.harness.complete_json", lambda *a, **k: {"invalid": True})
    graph = compile_agent(
        "TestAgent", "Test", {}, lambda c: {}, lambda c: result("complete", "Done")
    )
    output = graph.invoke(
        {
            "context": {
                "run_id": "run",
                "construction_events_path": str(tmp_path / "events"),
                "operation_directory": str(tmp_path / "ops"),
                "limits": {"max_failures": 2},
            }
        }
    )
    assert output["result"]["status"] == "blocked"
    assert output["step"] == 2


def test_sqlite_resume_keeps_completed_tool_and_excludes_human_wait(tmp_path, monkeypatch):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    clock = [0.0]
    monkeypatch.setattr("medbenchagent.harness.time.monotonic", lambda: clock[0])
    actions = iter(
        [
            {"tool": "compute", "arguments": {}, "reason": "Read input"},
            {"tool": "ask_user", "arguments": {"question": "Which language?"}, "reason": "Missing"},
            {"tool": "finish", "arguments": {}, "reason": "Complete"},
        ]
    )
    monkeypatch.setattr("medbenchagent.harness.complete_json", lambda *a, **k: next(actions))
    calls = []

    def compute(context):
        calls.append(1)
        return result("success", "Input examined")

    def build(connection):
        return compile_agent(
            "TestAgent",
            "Test",
            {"compute": compute},
            lambda c: c,
            lambda c: result("complete", "Done"),
            human=True,
            checkpointer=SqliteSaver(connection),
        )

    config = {"configurable": {"thread_id": "resume"}}
    context = {
        "run_id": "run",
        "user_requirement": "Build a benchmark",
        "construction_events_path": str(tmp_path / "events.jsonl"),
        "operation_directory": str(tmp_path / "ops"),
        "limits": {"max_seconds": 1},
    }
    with sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False) as connection:
        initial = build(connection).invoke({"context": context}, config)
        assert initial["__interrupt__"]
    clock[0] = 7200.0
    with sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False) as connection:
        resumed = build(connection).invoke(Command(resume="English"), config)
    assert calls == [1]
    assert resumed["result"]["status"] == "complete"
    assert resumed["context"]["user_requirement"].endswith("English")
    assert resumed["elapsed"] == 0
    assert resumed["observations"][0]["summary"] == "Input examined"
