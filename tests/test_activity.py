import json
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from medbenchagent.runtime.activity import ActivityReader
from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.native import InvocationGuard
from medbenchagent.settings import load_coding_settings


def test_tool_activity_arrives_before_completion_and_filters_sensitive_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "secret-test-value")
    entered, release = threading.Event(), threading.Event()
    request = SimpleNamespace(runtime=SimpleNamespace(context=SimpleNamespace(directory=tmp_path, runtime_directory=tmp_path,
        compilation_id="run", agent_id="data_processing", invocation_id="work", request=SimpleNamespace(operation="profile"))),
        state={"messages": []}, tool_call={"name": "inspect_sources", "id": "call", "args": {"api_key": "secret-test-value", "path": "secret-test-value"}})
    def slow(_):
        entered.set()
        assert release.wait(5)
        return ToolMessage(content="x" * 10000 + "secret-test-value", tool_call_id="call")
    thread = threading.Thread(target=lambda: InvocationGuard(None, 1000).wrap_tool_call(request, slow))
    thread.start()
    reader = ActivityReader(tmp_path)
    try:
        assert entered.wait(5)
        rows = reader.read()
        assert rows[-1]["event"]["event_type"] == "tool_started" and thread.is_alive()
        assert "secret-test-value" not in reader.path.read_text(encoding="utf-8")
    finally:
        release.set()
        thread.join(5)
    assert reader.read()[-1]["event"]["event_type"] == "tool_finished"
    assert reader.events[-1]["event"]["output"]["characters"] > 10000
    assert reader.read() == []
    pending = json.dumps({"event_id": "partial", "agent_id": "coding", "event_type": "coding_started"}).encode()
    with reader.path.open("ab") as output:
        output.write(pending[:20])
    assert reader.read() == []
    with reader.path.open("ab") as output:
        output.write(pending[20:] + b"\n")
    assert reader.read()[0]["id"] == "partial"
    other = ActivityReader(tmp_path / "other")
    record_event(other.path, "work_result", agent_id="review", outcome="completed", issues=[{"code": "review_template"}])
    assert other.read()[0]["action"] == "review · rework"
    assert len(reader.events) == 3


def test_native_sdk_callback_projects_actions_without_reasoning(tmp_path):
    python = load_coding_settings().worker_python
    if not Path(python).is_file():
        pytest.skip("Independent Coding environment is not provisioned")
    code = '''
import sys,json
from pathlib import Path
from medbenchagent.agents.data_processing.coding.worker import activity_callback
from openhands.sdk.event import ActionEvent,ObservationEvent
from openhands.sdk.llm import TextContent
from openhands.tools.terminal import TerminalAction,TerminalObservation
folder=sys.argv[1]
callback=activity_callback({"runtime_directory":folder,"conversation_id":"conversation","request":{"compilation_id":"run","request_id":"coding-call","workstream_id":"ws"},"parent_request_id":"dp-request","parent_operation":"profile"},["private-token"])
action=ActionEvent(thought=[TextContent(text="hidden-thought")],action=TerminalAction(command="echo private-token"),tool_name="terminal",tool_call_id="call",tool_call={"id":"call","name":"terminal","arguments":"{}","origin":"completion"},llm_response_id="response")
callback(action)
callback(ObservationEvent(tool_name="terminal",tool_call_id="call",action_id=action.id,observation=TerminalObservation.from_text("private-token result",command="echo private-token",exit_code=0)))
text=(Path(folder)/"events.jsonl").read_text(encoding="utf-8")
assert "hidden-thought" not in text and "private-token" not in text
events=[json.loads(line) for line in text.splitlines()]
assert [e["event_type"] for e in events]==["coding_action","coding_observation"]
assert all(e["parent_request_id"]=="dp-request" for e in events)
'''
    result = subprocess.run([python, "-c", code, str(tmp_path)], capture_output=True, text=True, encoding="utf-8", timeout=60,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src"), "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stderr
