"""Focused model-adapter coverage using the real structured-response repair path."""

from dataclasses import replace
import json
from types import SimpleNamespace

import httpx
from openai import APIConnectionError
import pytest

import medbenchagent.runtime.llm as transport
import medbenchagent.settings as settings_module
from medbenchagent.compilation.artifacts import write_json_artifact
from medbenchagent.compilation.contracts import AskUser, Delegate, Finish, WorkNeed, WorkRequest, WorkResult
from medbenchagent.compilation.graph import SupervisorView
from medbenchagent.compilation.supervisor import ModelSupervisor
from medbenchagent.settings import SupervisorSettings, load_supervisor_settings


def test_model_supervisor_repairs_choices_and_uses_bounded_frozen_context(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_module, "load_project_env", lambda: None)
    for name in settings_module._ENVIRONMENT_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MEDBENCH_CONFIG", raising=False)
    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "credential-not-for-snapshots")
    config = tmp_path / "model.toml"
    config.write_text(
        '[llm]\nmodel="frozen-model"\nbase_url="http://model.invalid/v1"\n'
        '[supervisor]\ncontext_chars=64000\nartifact_chars=40\nmax_artifacts=1\n'
        '[rag]\nchunk_size="not parsed by the Supervisor"\n'
        '[data_processing]\nbackend="not initialized"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        settings_module, "_settings_from",
        lambda *_args: pytest.fail("Supervisor must not parse full application settings"),
    )
    settings = load_supervisor_settings(config)
    snapshot = settings.snapshot()
    assert "credential-not-for-snapshots" not in json.dumps(snapshot)
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "changed-model")
    monkeypatch.setenv("MEDBENCH_LLM_BASE_URL", "http://changed.invalid/v1")
    settings = SupervisorSettings.from_snapshot(snapshot)
    assert settings.llm.model == "frozen-model"
    assert settings.llm.base_url == "http://model.invalid/v1"

    request = WorkRequest(
        request_id="model-input", compilation_id="model-test", agent_id="fixture",
        objective="Supply evidence",
    )
    evidence = write_json_artifact(
        tmp_path, request, "need.json", kind="data_requirement",
        value={"description": "evidence " * 100, "tail": "UNREAD_TAIL"},
    )
    plan = write_json_artifact(tmp_path, request, "plan.json", kind="task_plan", value={"ready": False})
    view = SupervisorView(
        compilation_id="model-test", goal="Inspect the available evidence",
        agents={"inspector": "Inspect supplied artifacts"}, artifacts=(evidence, plan),
        last_result=WorkResult(
            request_id=request.request_id, outcome="needs_input", summary="Need more evidence",
            artifacts=[plan],
            needs=[WorkNeed(need_id="data", kind="work", summary="Read the need", detail_ref=evidence)],
        ),
        history=(), answers=(),
    )
    requests = []
    answers = iter([
        {"decision": {"action": "delegate", "agent_id": "missing", "objective": "Inspect"}},
        {"decision": {"action": "delegate", "agent_id": "inspector", "objective": "Inspect",
                      "input_artifact_ids": ["invented"]}},
        {"decision": {"action": "delegate", "agent_id": "inspector", "objective": "Inspect",
                      "input_artifact_ids": [evidence.artifact_id]}},
        {"decision": {"action": "ask_user", "question": "Which output is required?"}},
        {"decision": {"action": "finish", "summary": "The requested inspection is complete"}},
    ])

    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs)))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(answers))))],
        )

    def client(frozen):
        assert frozen == settings.llm
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(transport, "_client", client)
    monkeypatch.setattr(transport, "get_settings", lambda: pytest.fail("Use explicit LLM settings"))
    monkeypatch.setattr(
        transport, "resolve_model_service",
        lambda *_args: pytest.fail("A frozen model must not be rediscovered on each call"),
    )
    supervisor = ModelSupervisor(tmp_path, settings)
    decision = supervisor(view)
    assert isinstance(decision, Delegate) and decision.inputs == [evidence]
    assert len(requests) == 3
    assert "Unregistered Agent" in requests[1]["messages"][-1]["content"]
    assert "Unknown artifact IDs" in requests[2]["messages"][-1]["content"]
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["artifact_previews"][0]["truncated"] is True
    assert len(payload["artifact_previews"][0]["content"]) == 40
    assert payload["omitted_preview_ids"] == [plan.artifact_id]
    assert "UNREAD_TAIL" not in json.dumps(payload)
    assert all("path" not in ref for ref in payload["available_artifacts"])
    assert all(item["model"] == "frozen-model" for item in requests)
    assert isinstance(supervisor(view), AskUser)
    assert isinstance(supervisor(view), Finish)

    answers = iter([{"decision": {"action": "delegate", "agent_id": "missing", "objective": "No"}}
                    for _ in range(3)])
    with pytest.raises(transport.StructuredOutputError):
        supervisor(view)
    count = len(requests)
    with pytest.raises(transport.RequestContextBudgetError):
        ModelSupervisor(tmp_path, replace(settings, context_chars=1))(view)
    assert len(requests) == count

    def unavailable(**_kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://model.invalid/v1"))

    monkeypatch.setattr(transport, "_client", lambda _settings: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=unavailable)),
    ))
    with pytest.raises(transport.ModelUnavailable):
        supervisor(view)
