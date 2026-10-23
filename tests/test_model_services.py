"""Critical discovery, credential, and frozen-identity behavior."""


import httpx
import pytest

from medbenchagent.runtime.model_services import (
    ModelDiscoveryError, resolve_model_service,
)


def test_discovery_requires_explicit_selection_for_multiple_models(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *_a, **_kw: httpx.Response(
        200, request=httpx.Request("GET", "http://chat/v1/models"),
        json={"data": [{"id": "a"}, {"id": "b"}]}))
    with pytest.raises(ModelDiscoveryError, match="multiple models"):
        resolve_model_service("Chat", "http://chat/v1", "", "key")
    assert resolve_model_service("Chat", "http://chat/v1", "b", "key") == "b"
    with pytest.raises(ModelDiscoveryError, match="unavailable"):
        resolve_model_service("Chat", "http://chat/v1", "missing", "key")
