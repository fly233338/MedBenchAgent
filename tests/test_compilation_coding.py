"""SDK candidate recovery remains independent from formal data publication."""
from runtime_fixtures import runtime_for
from dataclasses import replace
import sys
import pytest
from medbenchagent.agents.data_processing.coding.adapter import OpenHandsCodingAgent, run_local_worker
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.agents._shared.prompting import load_prompt
from medbenchagent.settings import load_coding_settings
from coding_fixtures import FakeRunner, Crash
from medbenchagent.agents.data_processing.coding.skills import freeze_skills


def test_worker_module_returns_structured_environment_failure_without_sdk_calls(tmp_path):
    request = WorkRequest(compilation_id="worker", request_id="one", agent_id="coding",
                          operation="execute", workstream_id="owned", objective="Check transport")
    candidate = run_local_worker({
        "request": request.model_dump(mode="json"),
        "settings": {"coding": {"worker_python": sys.executable, "timeout_seconds": 30,
                                 "sdk_packages": {"pydantic": "0.0.0"}}},
        "log_path": str(tmp_path / "worker.log"), "workspace": str(tmp_path),
    })
    assert candidate.request_id == request.request_id
    assert candidate.outcome == "blocked"
    assert candidate.issues[0].code == "environment_unavailable"
    assert "Frozen Coding dependency differs: pydantic" in candidate.issues[0].message


def test_workstreams_reuse_isolate_and_exchange_formal_artifacts(tmp_path):
    settings = load_coding_settings()
    settings = replace(settings, llm=replace(settings.llm, model="offline"))
    runner = FakeRunner()
    agent = OpenHandsCodingAgent(settings, instructions=load_prompt("data_processing/coding/prompts/base.md"), skills=freeze_skills(runtime_for(tmp_path), settings.skill_names), runner=runner)
    request = WorkRequest(compilation_id="sdk", request_id="one", agent_id="coding", operation="execute",
                          workstream_id="owned", objective="Write candidate")
    first, files = agent.execute_candidate(tmp_path, request, "Inspect", runtime_directory=runtime_for(tmp_path))
    assert first.outcome == "completed" and files
    again, repeated = agent.execute_candidate(tmp_path, request, "Inspect", runtime_directory=runtime_for(tmp_path))
    assert again == first and repeated == files
    assert not (tmp_path / "artifacts").exists()
    second, _ = agent.execute_candidate(tmp_path, request.model_copy(update={"request_id": "two"}), "Inspect", runtime_directory=runtime_for(tmp_path))
    assert second.request_id == "two"


def test_interrupted_sdk_turn_resumes_without_reexecuting_completed_code(tmp_path):
    settings = load_coding_settings()
    settings = replace(settings, llm=replace(settings.llm, model="offline"))
    request = WorkRequest(compilation_id="sdk", request_id="one", agent_id="coding", operation="execute",
                          workstream_id="owned", objective="Write candidate")
    broken = OpenHandsCodingAgent(settings, instructions=load_prompt("data_processing/coding/prompts/base.md"), skills=freeze_skills(runtime_for(tmp_path), settings.skill_names), runner=FakeRunner(crash="finished"))
    with pytest.raises(Crash):
        broken.execute_candidate(tmp_path, request, "Inspect", runtime_directory=runtime_for(tmp_path))
    runner = FakeRunner(lambda _: pytest.fail("Completed SDK code replayed"))
    restored = OpenHandsCodingAgent(settings, instructions=load_prompt("data_processing/coding/prompts/base.md"), skills=freeze_skills(runtime_for(tmp_path), settings.skill_names), runner=runner)
    result, files = restored.execute_candidate(tmp_path, request, "Inspect", runtime_directory=runtime_for(tmp_path))
    assert result.outcome == "completed" and files
