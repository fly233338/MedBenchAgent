from dataclasses import replace
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel

import medbenchagent.runtime.llm as llm
from medbenchagent.settings import get_settings


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
    with pytest.raises(llm.ModelUnavailable, match="failed validation"):
        llm.complete_validated_json("system", "user", _Decision)
    assert len(requests) == 2


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
    responses = iter(['{"ok":true}', '{"ok":true}', '[]', 'not json'])

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
        assert requests[0]["messages"] == [{"role": "system", "content": "system"},
                                           {"role": "user", "content": "user"}]
        with pytest.raises(llm.ModelUnavailable, match="must be an object"):
            llm.complete_json("system", "user")
        with pytest.raises(llm.ModelUnavailable, match="valid JSON"):
            llm.complete_json("system", "user")
    finally:
        llm.clear_client_cache()
