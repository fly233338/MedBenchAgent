import json

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
