import json
from pathlib import Path
from types import SimpleNamespace
from medbenchagent.construction import (
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
