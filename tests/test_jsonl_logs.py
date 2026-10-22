import json
import time
from types import SimpleNamespace

import pytest

from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.operation_log import append


def test_event_and_operation_logs_remain_append_only_jsonl(tmp_path):
    events = tmp_path / "events.jsonl"
    operations = tmp_path / "operations.jsonl"
    record_event(events, "first", value=1)
    record_event(events, "second", value=2)
    append(operations, {"id": "one", "status": "success"})
    append(operations, {"id": "two", "status": "success"})
    assert [json.loads(line)["event_type"] for line in events.read_text(encoding="utf-8").splitlines()] == ["first", "second"]
    assert [json.loads(line)["id"] for line in operations.read_text(encoding="utf-8").splitlines()] == ["one", "two"]


def test_explicit_model_logging_preserves_correlation_without_loading_app_settings(tmp_path, monkeypatch):
    import medbenchagent.runtime.construction as construction

    monkeypatch.setattr(construction, "get_settings", lambda: pytest.fail("Explicit model logging must stay isolated"))
    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "never-in-the-event-log")
    path = tmp_path / "events.jsonl"
    with construction.event_context(construction_events_path=str(path), compilation_id="comp",
                                    agent_id="supervisor", model="frozen-model", reasoning_effort="") as call_ids:
        construction.record_llm_call(status="success", started=time.perf_counter(),
                                     response=SimpleNamespace(id="provider-call", usage=None))
        construction.record_llm_validation_error(errors=["invalid choice"], model_output={"action": "invalid"})
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(row["compilation_id"] == "comp" and row["agent_id"] == "supervisor"
               and row["call_id"] == call_ids[0] and "request_id" not in row for row in records)
    assert records[0]["provider_response_id"] == "provider-call"
    assert "never-in-the-event-log" not in path.read_text(encoding="utf-8")
