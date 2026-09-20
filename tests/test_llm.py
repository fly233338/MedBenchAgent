from dataclasses import replace
from types import SimpleNamespace

import pytest

import medbenchagent.runtime.llm as llm
from medbenchagent.settings import get_settings


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
