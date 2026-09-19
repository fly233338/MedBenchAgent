import json

import pytest

from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.tool_contracts import NoArguments


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
        ("finish", {"approved": True}, "approved"),
    ],
)
def test_invalid_arguments_are_reported_before_execution(
    tmp_path, monkeypatch, tool, arguments, field
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

    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", decide)
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
    assert prompts[1]["last_result"]["errors"][0]["field"] == field
    assert prompts[0]["tools"]["profile"]["parameters"]["additionalProperties"] is False
    assert "finish" in prompts[0]["tools"]
    assert "ask_user" in prompts[0]["tools"]


def test_validated_defaults_reach_named_parameters(tmp_path, monkeypatch):
    actions = iter(
        [
            {"tool": "profile", "arguments": {}, "reason": "Profile"},
            {"tool": "finish", "arguments": {}, "reason": "Done"},
        ]
    )
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", lambda *a: next(actions))

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


@pytest.mark.parametrize("error", [KeyError("internal"), TypeError("internal")])
def test_internal_programming_errors_are_not_parameter_feedback(tmp_path, monkeypatch, error):
    monkeypatch.setattr(
        "medbenchagent.runtime.harness.complete_json",
        lambda *a: {
            "tool": "broken",
            "arguments": {},
            "reason": "Execute",
        },
    )

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
