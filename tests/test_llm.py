from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel

import medbenchagent.runtime.llm as llm
from medbenchagent.settings import get_settings


@pytest.fixture(autouse=True)
def no_live_discovery(monkeypatch):
    monkeypatch.setattr(llm, "resolve_model_service", lambda _kind, _url, model, _key: model or "test-chat")


class _Decision(BaseModel):
    kind: Literal["prepare_dataset"]


def test_validated_json_repairs_schema_error(monkeypatch):
    outputs = iter(['{"kind":"prepare"}', '{"kind":"prepare_dataset"}'])
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(outputs)))])

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result = llm.complete_validated_json("system", "user", _Decision)
    assert result.kind == "prepare_dataset"
    assert len(requests) == 2
    assert "prepare_dataset" in requests[1]["messages"][-1]["content"]


def test_validated_json_repairs_business_contract_error(monkeypatch):
    outputs = iter(['{"kind":"prepare_dataset"}', '{"kind":"prepare_dataset"}'])
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(outputs)))])

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    attempts = []

    def validate(_decision):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("tool arguments are invalid")

    llm.complete_validated_json("system", "user", _Decision, validate=validate)
    assert len(requests) == 2
    assert "tool arguments are invalid" in requests[1]["messages"][-1]["content"]


def test_validated_json_stops_repeated_error(monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"kind":"prepare"}'))])

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(llm.StructuredOutputError, match="failed validation"):
        llm.complete_validated_json("system", "user", _Decision)
    assert len(requests) == 2


def test_validation_failure_records_phase_schema_and_output(monkeypatch):
    diagnostics = []
    monkeypatch.setattr(llm, "record_llm_validation_error",
                        lambda **values: diagnostics.append(values))

    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"kind":"wrong"}'))])

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises(llm.StructuredOutputError):
        llm.complete_validated_json("system", "user", _Decision)
    assert diagnostics
    assert "schema=_Decision" in diagnostics[0]["errors"]
    assert diagnostics[0]["model_output"] == '{"kind":"wrong"}'


def test_validation_output_excerpt_preserves_head_and_tail():
    value = "a" * 3000
    excerpt = llm._validation_output_excerpt(value)
    assert len(excerpt) <= llm._VALIDATION_OUTPUT_LIMIT + 40
    assert excerpt.startswith("a" * 1600)
    assert excerpt.endswith("a" * 800)
    assert "chars truncated" in excerpt


def test_validated_json_streams_single_successful_call(monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
                    for piece in ('{"kind":', '"prepare_dataset"}'))

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    pieces = []
    assert llm.complete_validated_json("system", "user", _Decision, on_chunk=pieces.append).kind == "prepare_dataset"
    assert pieces == ['{"kind":', '"prepare_dataset"}']
    assert len(requests) == 1
    assert requests[0]["stream"] is True


def test_json_api_configuration_and_client_reuse(monkeypatch) -> None:
    configured = replace(get_settings().llm, base_url="http://127.0.0.1:9999/v1",
                         model="test-model", timeout_seconds=17, max_retries=2)
    captured = {}
    requests = []
    responses = iter(['{"ok":true}', '{"ok":true}', '[]', '{"ok":true}'])

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(llm=configured))
    monkeypatch.setattr(llm, "get_llm_api_key", lambda: None)
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "wrap_openai", lambda client, **kwargs: client)
    llm.clear_client_cache()
    try:
        assert llm.complete_json("system", "user") == {"ok": True}
        client = llm._client()
        assert llm.complete_json("system", "user") == {"ok": True}
        assert llm._client() is client
        assert captured["api_key"]
        assert captured["base_url"] == configured.base_url
        assert captured["timeout"] == configured.timeout_seconds
        assert captured["max_retries"] == configured.max_retries
        assert requests[0]["model"] == configured.model
        assert requests[0]["response_format"] == {"type": "json_object"}
        assert "schema" in requests[0]["messages"][0]["content"]
        assert requests[0]["messages"][1] == {"role": "user", "content": "user"}
        assert llm.complete_json("system", "user") == {"ok": True}
        assert len(requests) == 4
    finally:
        llm.clear_client_cache()


@pytest.mark.parametrize("choices", [[], [SimpleNamespace(message=SimpleNamespace(content=""))],
                                    [SimpleNamespace(message=SimpleNamespace(content="not json"))]])
def test_empty_or_invalid_response_is_repaired(monkeypatch, choices):
    responses = iter([SimpleNamespace(choices=choices), SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content='{"kind":"prepare_dataset"}'))])])
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kwargs: next(responses)))))
    assert llm.complete_validated_json("system", "user", _Decision).kind == "prepare_dataset"


def test_config_switch_does_not_close_active_clients(monkeypatch):
    configured = get_settings().llm
    closed = []
    clients = []

    def create(**kwargs):
        client = SimpleNamespace(close=lambda: closed.append(kwargs["base_url"]))
        clients.append(client)
        return client

    llm.clear_client_cache()
    monkeypatch.setattr(llm, "OpenAI", create)
    monkeypatch.setattr(llm, "wrap_openai", lambda client, **kwargs: client)
    monkeypatch.setattr(llm, "get_llm_api_key", lambda: "test")
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(llm=configured))
    try:
        first = llm._client()
        configured = replace(configured, base_url="http://other.invalid/v1")
        assert llm._client() is not first
        assert not closed
        assert len(clients) == 2
    finally:
        llm.clear_client_cache()


def test_budget_covers_entire_serialized_request_including_schema_and_tools(monkeypatch):
    requests = []
    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"kind":"prepare_dataset"}'))])
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    prompt = json.dumps({"tools": {"inspect": {"parameters": {"description": "x" * 300}}},
                         "evidence": "医学证据" * 100, "last_result": {"data": "y" * 100}}, ensure_ascii=False)
    llm.complete_json("required instructions", prompt, _Decision)
    size = len(json.dumps(requests[0], ensure_ascii=False))
    assert size > len(prompt)
    llm.complete_json("required instructions", prompt, _Decision, request_char_budget=size)
    with pytest.raises(llm.RequestContextBudgetError) as error:
        llm.complete_json("required instructions", prompt, _Decision, request_char_budget=size - 1)
    assert len(requests) == 2
    assert error.value.diagnostics == {"request_chars": size, "context_chars": size - 1}


def test_repair_feedback_is_rebudgeted_without_truncating_model_output(monkeypatch):
    requests = []
    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="invalid " + "x" * 7000))])
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    with pytest.raises(llm.RequestContextBudgetError) as error:
        llm.complete_validated_json("system", "evidence", _Decision, request_char_budget=1000)
    assert len(requests) == 1
    assert error.value.diagnostics["request_chars"] > 7000


def test_direct_planning_calls_use_the_configured_final_request_budget(monkeypatch):
    from medbenchagent.planning.context import complete_planning_json
    from medbenchagent.settings import active_settings
    settings = get_settings()
    monkeypatch.setattr(llm, "_client", lambda: pytest.fail("Over-budget request must not reach the model"))
    with active_settings(replace(settings, planning=replace(settings.planning, context_chars=1))):
        with pytest.raises(llm.RequestContextBudgetError):
            complete_planning_json("system", "required source evidence", _Decision)
