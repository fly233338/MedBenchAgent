import json
from pathlib import Path

from medbenchagent.construction import construction_report, record_event


def test_construction_events_are_append_only_metadata_and_report_missing_usage(tmp_path: Path) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    record_event(
        events_path,
        "llm_call",
        run_id="run",
        dataset_id="vindr-mammo",
        task_id="view",
        agent="deepseek",
        operation="generation",
        status="success",
        latency_ms=12,
        input_tokens=0,
        output_tokens=0,
        token_usage_available=False,
        call_id="call-1",
    )
    record_event(events_path, "review_accept", run_id="run", dataset_id="vindr-mammo", task_id="view")

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert all("prompt" not in event and "response" not in event for event in events)
    report = construction_report(events_path)
    assert report["llm_calls"] == 1
    assert report["token_usage_complete"] is False
    assert report["retention"] == 1.0
