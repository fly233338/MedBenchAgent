from langgraph.checkpoint.memory import MemorySaver
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.workflow.graph import compile_benchmark_graph


def test_supervisor_replaces_fixed_role_chain():
    graph = compile_benchmark_graph().get_graph()
    assert set(graph.nodes) == {"__start__", "initialize", "supervisor", "__end__"}


def test_intent_gates_preparation_and_benchmark_stages(tmp_path):
    from medbenchagent.workflow.graph import set_intent, available_stages
    from medbenchagent.runtime.dependencies import publish
    context = {"run_directory": str(tmp_path)}
    assert available_stages(context) == {"set_intent", "ask_user", "list_datasets", "inspect_dataset_status"}
    context.update(set_intent(context, intent="prepare_dataset", prepare_dataset_ids=["new"])["updates"])
    assert "prepare_data" in available_stages(context)
    assert "specify" not in available_stages(context)
    publish(context, "dataset/new/Dataset", {}, ())
    context["prepared_entries"] = {"new": {}}
    assert "finish" in available_stages(context)
    context.update(set_intent(context, intent="prepare_then_build", prepare_dataset_ids=["new"])["updates"])
    assert "specify" in available_stages(context)
    assert "finish" not in available_stages(context)


def test_harness_selects_tools_and_finishes_without_confirmation(tmp_path, monkeypatch):
    actions = iter(
        [
            {"tool": "compute", "arguments": {}, "reason": "Needed"},
            {"tool": "finish", "arguments": {}, "reason": "Done"},
        ]
    )
    monkeypatch.setattr(
        "medbenchagent.runtime.harness.complete_json", lambda *a, **k: next(actions)
    )
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
    monkeypatch.setattr(
        "medbenchagent.runtime.harness.complete_json", lambda *a, **k: {"invalid": True}
    )
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
    monkeypatch.setattr("medbenchagent.runtime.harness.time.monotonic", lambda: clock[0])
    actions = iter(
        [
            {"tool": "compute", "arguments": {}, "reason": "Read input"},
            {"tool": "ask_user", "arguments": {"question": "Which language?"}, "reason": "Missing"},
            {"tool": "finish", "arguments": {}, "reason": "Complete"},
        ]
    )
    monkeypatch.setattr(
        "medbenchagent.runtime.harness.complete_json", lambda *a, **k: next(actions)
    )
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
    assert resumed["context"]["clarification"]["answer"] == "English"
    assert resumed["elapsed"] == 0
    assert resumed["observations"][0]["summary"] == "Input examined"


def test_dynamic_catalog_and_execution_use_same_gate(tmp_path, monkeypatch):
    import json
    prompts = []
    actions = iter(["guarded", "finish"])
    def model(_, prompt):
        prompts.append(json.loads(prompt))
        return {"tool": next(actions), "arguments": {}, "reason": "Attempt"}
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", model)
    calls = []
    graph = compile_agent("Gate", "Gate", {"guarded": lambda c: calls.append(1)}, lambda c: {},
                          lambda c: result("complete", "Done"), available_tools=lambda c: {"finish", "ask_user"})
    graph.invoke({"context": {"run_id": "gate", "operation_directory": str(tmp_path / "ops"),
                              "construction_events_path": str(tmp_path / "events")}})
    assert not calls
    assert "guarded" not in prompts[0]["tools"]
    assert prompts[1]["last_result"]["status"] == "failed"


def test_global_needs_input_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", lambda *args: {
        "tool": "ask_user", "arguments": {"question": "Clarify target", "scope": "global"}, "reason": "Ambiguous"})
    graph = compile_agent("Child", "Child", {}, lambda c: {}, lambda c: result("complete", "Done"))
    output = graph.invoke({"context": {"run_id": "input", "operation_directory": str(tmp_path / "ops"),
                                       "construction_events_path": str(tmp_path / "events")}})
    assert output["result"]["status"] == "needs_input"
    assert output["result"]["unresolved_questions"] == ["Clarify target"]
