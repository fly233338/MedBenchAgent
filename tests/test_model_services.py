"""Critical discovery, credential, and frozen-identity behavior."""

from dataclasses import replace
from urllib.parse import urlsplit

import httpx
import pytest

from medbenchagent.runtime.model_services import (
    ModelDiscoveryError, resolve_model_service, resolve_run_services, verify_frozen_services,
)
from medbenchagent.settings import load_settings


def test_discovery_requires_explicit_selection_for_multiple_models(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *_a, **_kw: httpx.Response(
        200, request=httpx.Request("GET", "http://chat/v1/models"),
        json={"data": [{"id": "a"}, {"id": "b"}]}))
    with pytest.raises(ModelDiscoveryError, match="multiple models"):
        resolve_model_service("Chat", "http://chat/v1", "", "key")
    assert resolve_model_service("Chat", "http://chat/v1", "b", "key") == "b"
    with pytest.raises(ModelDiscoveryError, match="unavailable"):
        resolve_model_service("Chat", "http://chat/v1", "missing", "key")


def test_three_credentials_are_independent_and_secrets_are_not_frozen(monkeypatch):
    settings = load_settings()
    keys = {"8005": "chat-key", "8006": "embedding-key", "8007": "reranker-key"}
    models = {"8005": "chat", "8006": "embedding", "8007": "reranker"}
    for kind, value in (("LLM", keys["8005"]), ("EMBEDDING", keys["8006"]),
                        ("RERANKER", keys["8007"])):
        monkeypatch.setenv(f"MEDBENCH_{kind}_API_KEY", value)
    seen = []

    def get(url, *, headers, **_kwargs):
        port = str(urlsplit(url).port)
        seen.append((port, headers["Authorization"]))
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"data": [{"id": models[port]}]})

    monkeypatch.setattr(httpx, "get", get)
    frozen = resolve_run_services(settings)
    assert seen == [(port, f"Bearer {key}") for port, key in keys.items()]
    assert (frozen.llm.model, frozen.rag.embedding_model, frozen.rag.reranker_model) == (
        "chat", "embedding", "reranker")
    assert all(key not in str(frozen.snapshot()) for key in keys.values())

    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "rotated-chat-key")
    verify_frozen_services(frozen)
    assert seen[3] == ("8005", "Bearer rotated-chat-key")
    models["8005"] = "different"
    with pytest.raises(ModelDiscoveryError, match="unavailable"):
        verify_frozen_services(frozen)


def test_unavailable_rag_is_frozen_off_for_run(monkeypatch):
    settings = load_settings()

    def get(url, **_kwargs):
        if urlsplit(url).port == 8005:
            return httpx.Response(200, request=httpx.Request("GET", url),
                                  json={"data": [{"id": "chat"}]})
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", get)
    frozen = resolve_run_services(replace(settings, rag=replace(
        settings.rag, embedding_model="", reranker_model="")))
    assert frozen.llm.model == "chat"
    assert frozen.rag.embedding_model == frozen.rag.reranker_model == ""
    assert frozen.service_status["embedding"] != "available"
    assert frozen.service_status["reranker"] != "available"
    verify_frozen_services(frozen)
