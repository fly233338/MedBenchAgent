import json
from pathlib import Path
from types import SimpleNamespace

from medbenchagent.construction import construction_report, event_context, record_event, record_llm_call


def test_construction_events_are_append_only_metadata_and_report_missing_usage(tmp_path: Path) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    record_event(
        events_path,
        "llm_call",
        run_id="run",
        dataset_id="vindr-mammo",
        task_id="view",
        agent="SpecificationAgent",
        operation="generation",
        status="success",
        latency_ms=12,
        input_tokens=0,
        output_tokens=0,
        token_usage_available=False,
        call_id="call-1",
    )
    record_event(events_path, "run_started", run_id="run")
    record_event(events_path, "generation_summary", task_counts={"view": 1}, shortfalls={"view": 0})
    record_event(events_path, "audit_summary", audited=1, rejected=0)
    record_event(
        events_path,
        "human_action",
        stage="materials",
        action="confirm",
        modified_objects=2,
    )
    record_event(events_path, "run_completed", run_id="run")

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 6
    assert all("prompt" not in event and "response" not in event for event in events)
    report = construction_report(events_path)
    assert report["llm"]["calls"] == 1
    assert report["llm"]["per_audited_item"] == {"calls": 1, "tokens": 0}
    assert report["items"] == {"target": 1, "raw": 1, "audited": 1, "rejected": 0, "shortfall": 0}
    assert report["agents"]["SpecificationAgent"]["calls"] == 1
    assert report["human_stages"]["materials"] == {
        "confirmed": 1,
        "revised": 0,
        "revision_rounds": 0,
        "modified_objects": 2,
    }
    assert report["human_stages"]["audit_override"]["modified_objects"] == 0


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
