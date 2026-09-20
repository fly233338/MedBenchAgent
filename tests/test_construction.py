import json
from pathlib import Path
from types import SimpleNamespace
from medbenchagent.runtime.construction import (
    event_context,
    record_llm_call,
)


def test_llm_events_include_contextual_model_and_component(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    with event_context(
        construction_events_path=events_path,
        run_id="run",
        agent="GenerationAgent",
        operation="generate",
        component="letter",
    ):
        record_llm_call(
            status="success",
            started=0,
            response=SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3), id="response-1"
            ),
        )
    event = json.loads(events_path.read_text(encoding="utf-8"))
    assert event["agent"] == "GenerationAgent"
    assert event["operation"] == "generate"
    assert event["component"] == "letter"
    assert event["model"] == "deepseek-v4-flash"


def test_event_without_agent_does_not_invent_provider_role(tmp_path):
    from medbenchagent.runtime.construction import record_llm_validation_error
    path = tmp_path / "events.jsonl"
    with event_context(construction_events_path=path, run_id="run"):
        record_llm_call(status="success", started=0)
        record_llm_validation_error(errors=["invalid shape"], model_output=[])
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all("agent" not in event for event in events)
    assert events[0]["call_id"] == events[1]["call_id"]
