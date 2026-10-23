"""SDK candidate recovery remains independent from formal data publication."""
from dataclasses import replace
import pytest
from medbenchagent.compilation.coding import OpenHandsCodingAgent
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.prompts import load_prompt
from medbenchagent.settings import load_coding_settings
from coding_fixtures import FakeRunner, Crash


def test_workstreams_reuse_isolate_and_exchange_formal_artifacts(tmp_path):
    settings = load_coding_settings()
    settings = replace(settings, llm=replace(settings.llm, model="offline"))
    runner = FakeRunner()
    agent = OpenHandsCodingAgent(settings, instructions=load_prompt("coding.md"), runner=runner)
    request = WorkRequest(compilation_id="sdk", request_id="one", agent_id="coding", operation="execute",
                          workstream_id="owned", objective="Write candidate")
    first, files = agent.execute_candidate(tmp_path, request, "Inspect")
    assert first.outcome == "completed" and files
    again, repeated = agent.execute_candidate(tmp_path, request, "Inspect")
    assert again == first and repeated == files
    assert not (tmp_path / "artifacts").exists()
    second, _ = agent.execute_candidate(tmp_path, request.model_copy(update={"request_id": "two"}), "Inspect")
    assert second.request_id == "two"


def test_interrupted_sdk_turn_resumes_without_reexecuting_completed_code(tmp_path):
    settings = load_coding_settings()
    settings = replace(settings, llm=replace(settings.llm, model="offline"))
    request = WorkRequest(compilation_id="sdk", request_id="one", agent_id="coding", operation="execute",
                          workstream_id="owned", objective="Write candidate")
    broken = OpenHandsCodingAgent(settings, instructions=load_prompt("coding.md"), runner=FakeRunner(crash="finished"))
    with pytest.raises(Crash):
        broken.execute_candidate(tmp_path, request, "Inspect")
    runner = FakeRunner(lambda _: pytest.fail("Completed SDK code replayed"))
    restored = OpenHandsCodingAgent(settings, instructions=load_prompt("coding.md"), runner=runner)
    result, files = restored.execute_candidate(tmp_path, request, "Inspect")
    assert result.outcome == "completed" and files
