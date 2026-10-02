import json

import pytest

from medbenchagent.runtime.harness import (
    ToolAction,
    _EVENT_SUMMARY_LIMIT,
    _event_summary_excerpt,
    compile_agent,
    result,
)
from medbenchagent.runtime.clarifications import resolve_subject, validate_ask_user
from medbenchagent.runtime.tool_contracts import AskUserArguments, NoArguments


def test_event_summary_excerpt_preserves_short_and_long_values():
    assert _event_summary_excerpt("short") == "short"
    value = "a" * (_EVENT_SUMMARY_LIMIT + 500) + "TAIL"
    excerpt = _event_summary_excerpt(value)
    assert "chars truncated" in excerpt
    assert excerpt.startswith("a" * 1200)
    assert excerpt.endswith("a" * 596 + "TAIL")
    assert len(excerpt) <= _EVENT_SUMMARY_LIMIT + 40


def test_tool_action_schema_describes_outer_reason_contract():
    properties = ToolAction.model_json_schema()["properties"]
    assert "registered tool" in properties["tool"]["description"]
    assert "selected tool" in properties["arguments"]["description"]
    assert "outer ToolAction" in properties["reason"]["description"]


class ProfileArguments(NoArguments):
    revision: str = "1"
    refresh: bool = False


@pytest.mark.parametrize(
    "tool,arguments,field",
    [
        ("profile", {"refresh": "false"}, "refresh"),
        ("profile", {"extra": 1}, "extra"),
        ("ask_user", {}, "question"),
        ("ask_user", {"question": "  "}, "question"),
        ("ask_user", {"question": "Compare labels", "subject_scope": "multi_source"}, "source_ids"),
        ("finish", {"approved": True}, "approved"),
    ],
)
def test_invalid_arguments_are_reported_before_execution(
    tmp_path, monkeypatch, harness_model, tool, arguments, field
):
    prompts = []
    actions = iter(
        [
            {"tool": tool, "arguments": arguments, "reason": "Try"},
            {"tool": "finish", "arguments": {}, "reason": "Done"},
        ]
    )

    def decide(instructions, prompt):
        prompts.append(json.loads(prompt))
        return next(actions)

    adapter = harness_model(decide)
    calls = []
    graph = compile_agent(
        "Test",
        "Test",
        {"profile": lambda context, **kwargs: calls.append(kwargs)},
        lambda context: {},
        lambda context: result("complete", "Done"),
        args_schemas={"profile": ProfileArguments},
    )
    output = graph.invoke(
        {
            "context": {
                "run_id": "test",
                "operation_directory": str(tmp_path / "ops"),
                "construction_events_path": str(tmp_path / "events.jsonl"),
            }
        }
    )
    assert output["result"]["status"] == "complete"
    assert not calls
    assert len(prompts) == 2
    assert field in adapter.validation_errors[0]
    assert output["repair_attempts"] == 1
    assert output["execution_failures"] == 0
    assert prompts[0]["tools"]["profile"]["parameters"]["additionalProperties"] is False
    assert "finish" in prompts[0]["tools"]
    assert "ask_user" in prompts[0]["tools"]


def test_validated_defaults_reach_named_parameters(tmp_path, monkeypatch, harness_model):
    actions = iter(
        [
            {"tool": "profile", "arguments": {}, "reason": "Profile"},
            {"tool": "finish", "arguments": {}, "reason": "Done"},
        ]
    )
    harness_model(lambda *a: next(actions))

    def profile(context, *, revision, refresh):
        assert revision == "1"
        assert refresh is False
        return result("success", "Profiled", profiled=True)

    graph = compile_agent(
        "Test",
        "Test",
        {"profile": profile},
        lambda context: {},
        lambda context: result("complete", "Done"),
        args_schemas={"profile": ProfileArguments},
    )
    output = graph.invoke(
        {
            "context": {
                "run_id": "test",
                "operation_directory": str(tmp_path / "ops"),
                "construction_events_path": str(tmp_path / "events.jsonl"),
            }
        }
    )
    assert output["context"]["profiled"]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ("Test", "decide") in {(item.get("agent"), item.get("operation")) for item in events
                                 if item["event_type"] == "agent_step_started"}
    assert ("Test", "profile") in {(item.get("agent"), item.get("operation")) for item in events
                                  if item["event_type"] == "tool_started"}


def test_source_clarification_is_repaired_before_execution(tmp_path, harness_model):
    actions = iter([
        {"tool": "ask_user", "arguments": {"question": "What does status 0 mean?",
                                         "subject_scope": "source", "source_ids": ["one"]}, "reason": "Clarify"},
        {"tool": "finish", "arguments": {}, "reason": "Stop"},
    ])
    adapter = harness_model(lambda *_args: next(actions))
    graph = compile_agent("DataProcessingAgent", "Test", {}, lambda _context: {},
                          lambda _context: result("complete", "Done"))
    output = graph.invoke({"context": {"run_id": "test", "sources": [{"id": "one"}],
                                       "operation_directory": str(tmp_path / "ops"),
                                       "construction_events_path": str(tmp_path / "events.jsonl")}})
    assert output["result"]["status"] == "complete"
    assert output["repair_attempts"] == 1
    assert output["execution_failures"] == 0
    assert "Select the source" in adapter.validation_errors[0]
    assert resolve_subject({"source_scope": "one", "sources": [{"id": "one"}]},
                           "DataProcessingAgent", "source", []) == ("source", ["one"])


def test_supervisor_can_follow_up_on_child_issue():
    issue = {"issue_id": "issue-1", "role": "DataProcessingAgent"}
    arguments = AskUserArguments(question="Which code applies?", issue_id="issue-1").model_dump()
    validate_ask_user({"global_issue": issue}, "SupervisorAgent", arguments)


@pytest.mark.parametrize("error", [KeyError("internal"), TypeError("internal")])
def test_internal_programming_errors_are_not_parameter_feedback(tmp_path, monkeypatch, harness_model, error):
    harness_model(lambda *a: {
        "tool": "broken",
        "arguments": {},
        "reason": "Execute",
    })

    def broken(context):
        raise error

    graph = compile_agent(
        "Test",
        "Test",
        {"broken": broken},
        lambda context: {},
        lambda context: result("complete", "Done"),
    )
    with pytest.raises(type(error), match="internal"):
        graph.invoke(
            {
                "context": {
                    "run_id": "test",
                    "operation_directory": str(tmp_path / "ops"),
                    "construction_events_path": str(tmp_path / "events.jsonl"),
                }
            }
        )
