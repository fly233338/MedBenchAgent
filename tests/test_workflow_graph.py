from langgraph.checkpoint.memory import MemorySaver
from medbenchagent.harness import compile_agent, result
from medbenchagent.workflow.graph import compile_benchmark_graph

def test_supervisor_replaces_fixed_role_chain():
    graph = compile_benchmark_graph().get_graph()
    assert set(graph.nodes) == {"__start__", "initialize", "supervisor", "__end__"}

def test_harness_selects_tools_and_finishes_without_confirmation(tmp_path, monkeypatch):
    actions = iter([{"tool": "compute", "arguments": {}, "reason": "Needed"},
                    {"tool": "finish", "arguments": {}, "reason": "Done"}])
    monkeypatch.setattr("medbenchagent.harness.complete_json", lambda *a, **k: next(actions))
    calls = []
    def compute(context, args):
        calls.append(1)
        return result("success", "Computed", value=3)
    graph = compile_agent("TestAgent", "Test", {"compute": compute}, lambda c: c,
                          lambda c, a: result("complete", "Done"), checkpointer=MemorySaver())
    context = {"run_id": "run", "construction_events_path": str(tmp_path / "events.jsonl"),
               "operation_directory": str(tmp_path / "ops")}
    output = graph.invoke({"context": context}, {"configurable": {"thread_id": "test"}})
    assert output["context"]["value"] == 3
    assert output["result"]["status"] == "complete"
    assert calls == [1]

def test_harness_stops_repeated_invalid_actions(tmp_path, monkeypatch):
    monkeypatch.setattr("medbenchagent.harness.complete_json", lambda *a, **k: {"invalid": True})
    graph = compile_agent("TestAgent", "Test", {}, lambda c: {}, lambda c, a: result("complete", "Done"))
    output = graph.invoke({"context": {"run_id": "run", "construction_events_path": str(tmp_path / "events"),
        "operation_directory": str(tmp_path / "ops"), "limits": {"max_failures": 2}}})
    assert output["result"]["status"] == "blocked"
    assert output["step"] == 2

