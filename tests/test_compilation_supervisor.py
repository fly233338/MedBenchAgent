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
from medbenchagent.compilation.contracts import (
    AskUser, Delegate, Finish, WorkNeed, WorkRecord, WorkRequest, WorkResult, Workstream,
)
from medbenchagent.compilation.interfaces import SupervisorView
from medbenchagent.prompts import load_prompt
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
    instructions = load_prompt("supervisor.md")
    supervisor = ModelSupervisor(tmp_path, settings, instructions=instructions)
    decision = supervisor(view)
    assert isinstance(decision, Delegate) and decision.inputs == [evidence]
    assert len(requests) == 3
    assert "Unregistered Agent" in requests[1]["messages"][-1]["content"]
    assert "Unknown artifact IDs" in requests[2]["messages"][-1]["content"]
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["artifact_previews"][0]["truncated"] is True
    assert len(payload["artifact_previews"][0]["content"]) == 40
    assert payload["omitted_counts"]["previews"] == 1
    assert "omitted_preview_ids" not in payload
    assert "UNREAD_TAIL" not in json.dumps(payload)
    assert all("path" not in ref for ref in payload["available_artifacts"])
    assert all(item["model"] == "frozen-model" for item in requests)
    assert isinstance(supervisor(view), AskUser)
    assert isinstance(supervisor(view), Finish)

    streams = (
        Workstream(workstream_id="ws-inspect", agent_id="inspector", objective="Inspect evidence", scope=["same"]),
        Workstream(workstream_id="ws-other", agent_id="other", objective="Other work", scope=["same"]),
    )
    scoped = replace(view, workstreams=streams, workstream_agents=("inspector", "other"),
                     history=(WorkRecord(request=WorkRequest(request_id="previous", compilation_id=view.compilation_id,
                               agent_id="inspector", workstream_id="ws-inspect", objective="Inspect"),
                               result=WorkResult(request_id="previous", outcome="completed", summary="Inspected")),))
    answers = iter([
        {"decision": {"action": "delegate", "agent_id": "inspector", "objective": "Continue",
                      "workstream_id": "unknown"}},
        {"decision": {"action": "delegate", "agent_id": "inspector", "objective": "Continue",
                      "workstream_id": "ws-other"}},
        {"decision": {"action": "delegate", "agent_id": "inspector", "objective": "Continue",
                      "workstream_id": "ws-inspect"}},
        {"decision": {"action": "delegate", "agent_id": "inspector", "objective": "Independent work",
                      "workstream_id": None}},
    ])
    decision = supervisor(scoped)
    assert decision.workstream_id == "ws-inspect"
    assert "Unknown Workstream" in requests[-2]["messages"][-1]["content"]
    assert "another Agent" in requests[-1]["messages"][-1]["content"]
    assert supervisor(scoped).workstream_id is None
    payload = json.loads(requests[-1]["messages"][1]["content"])
    assert payload["workstreams"][0]["last_result"]["summary"] == "Inspected"
    assert "conversation_id" not in json.dumps(payload)
    with pytest.raises(ValueError, match="does not use Workstreams"):
        replace(scoped, workstream_agents=()).validate_workstream("inspector", "ws-inspect")

    answers = iter([{"decision": {"action": "delegate", "agent_id": "missing", "objective": "No"}}
                    for _ in range(3)])
    with pytest.raises(transport.StructuredOutputError):
        supervisor(view)
    count = len(requests)
    with pytest.raises(transport.RequestContextBudgetError):
        ModelSupervisor(tmp_path, replace(settings, context_chars=1), instructions=instructions)(view)
    assert len(requests) == count

    def unavailable(**_kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://model.invalid/v1"))

    monkeypatch.setattr(transport, "_client", lambda _settings: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=unavailable)),
    ))
    with pytest.raises(transport.ModelUnavailable):
        supervisor(view)


def test_bounded_context_inspects_old_artifacts_and_preserves_reported_feedback(tmp_path, monkeypatch):
    from medbenchagent.compilation.supervisor import SupervisorLookupLimit

    monkeypatch.setattr(settings_module, "load_project_env", lambda: None)
    for name in (*settings_module._ENVIRONMENT_OVERRIDES, "MEDBENCH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    configured = load_supervisor_settings()
    settings = replace(configured, llm=replace(configured.llm, model="offline"), context_chars=14000,
                       catalog_page_size=1)
    first = WorkRequest(request_id="old", compilation_id="long", agent_id="inspector",
                        workstream_id="ws-0", objective="Create the earlier evidence")
    old = write_json_artifact(tmp_path, first, "old.json", kind="legacy-note", value={"evidence": "HISTORICAL"})
    refs, history, streams = [], [], []
    for index in range(1000):
        ref = old if index == 0 else old.model_copy(update={
            "artifact_id": f"artifact-{index}", "kind": "recent-note", "path": f"artifacts/r-{index}/note.json",
        })
        request = first if index == 0 else WorkRequest(
            request_id=f"r-{index}", compilation_id="long", agent_id="inspector",
            workstream_id=f"ws-{index}", objective="Later work " * 60,
        )
        result = WorkResult(request_id=request.request_id, outcome="needs_input" if index == 0 else "completed",
                            summary="A long work summary " * 100, artifacts=[ref],
                            needs=[WorkNeed(need_id="old-need", kind="work", summary="Check the old evidence",
                                            detail_ref=old)] if index == 0 else [])
        refs.append(ref)
        history.append(WorkRecord(request=request, result=result))
        streams.append(Workstream(workstream_id=f"ws-{index}", agent_id="inspector",
                                  objective="Independent work " * 50, scope=[f"scope-{index}"]))
    view = SupervisorView(compilation_id="long", goal="Find and inspect the earlier evidence",
                          agents={"inspector": "Inspect evidence"}, artifacts=tuple(refs), last_result=history[-1].result,
                          history=tuple(history), answers=("Retain my original constraint",),
                          workstreams=tuple(streams), workstream_agents=("inspector",))
    answers = iter([
        {"action": "inspect", "target": "work", "feedback_only": True},
        {"action": "inspect", "target": "artifacts", "kind": "legacy-note", "request_id": "old", "limit": 1},
        {"action": "inspect", "target": "preview", "ids": [old.artifact_id]},
        {"action": "delegate", "agent_id": "inspector", "workstream_id": "ws-0", "objective": "Verify old evidence",
         "input_artifact_ids": [old.artifact_id]},
    ])
    requests = []

    def create(**kwargs):
        assert len(json.dumps(kwargs, ensure_ascii=False)) <= settings.context_chars
        requests.append(json.loads(kwargs["messages"][1]["content"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"decision": next(answers)}),
        ))])

    monkeypatch.setattr(transport, "_client", lambda _: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    supervisor = ModelSupervisor(tmp_path, settings, instructions=load_prompt("supervisor.md"))
    decision = supervisor(view)
    assert decision.inputs == [old] and decision.workstream_id == "ws-0"
    assert requests[0]["goal"] == view.goal and requests[0]["user_answers"] == list(view.answers)
    assert requests[0]["omitted_counts"]["history"] > 0 and requests[0]["omitted_counts"]["artifacts"] > 0
    assert old.artifact_id not in {row["artifact_id"] for row in requests[0]["available_artifacts"]}
    feedback = requests[1]["inspections"][-1]["items"][0]
    assert feedback["need_count"] == 1 and feedback["needs"][0]["need_id"] == "old-need"
    assert requests[2]["inspections"][-1]["items"][0]["artifact_id"] == old.artifact_id
    assert "HISTORICAL" in requests[3]["inspections"][-1]["items"][0]["content"]
    assert all("omitted_preview_ids" not in payload for payload in requests)
    count = len(requests)
    with pytest.raises(transport.RequestContextBudgetError):
        supervisor(replace(view, answers=("Must not truncate this user answer " * 1000,)))
    assert len(requests) == count
    answers = iter([{"action": "inspect", "target": "artifacts", "limit": 1}] * (settings.max_lookups + 1))
    with pytest.raises(SupervisorLookupLimit):
        supervisor(view)
