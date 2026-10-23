"""Resolve OpenAI-compatible model identities without persisting credentials."""

from __future__ import annotations


import httpx



class ModelDiscoveryError(RuntimeError):
    """A model endpoint cannot provide the requested identity."""


def resolve_model_service(kind: str, base_url: str, explicit_model: str, api_key: str | None) -> str:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=15)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        detail = (f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError)
                  else type(exc).__name__)
        raise ModelDiscoveryError(f"{kind} model discovery failed: {detail}") from exc
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or any(not isinstance(item, dict) or
                                         not isinstance(item.get("id"), str) or not item["id"]
                                         for item in data):
        raise ModelDiscoveryError(f"{kind} /models returned invalid data")
    models = [item["id"] for item in data]
    if not models or len(models) != len(set(models)):
        raise ModelDiscoveryError(f"{kind} /models returned no unique models")
    if explicit_model:
        if explicit_model not in models:
            raise ModelDiscoveryError(f"{kind} model {explicit_model!r} is unavailable")
        return explicit_model
    if len(models) != 1:
        raise ModelDiscoveryError(f"{kind} endpoint has multiple models; configure an explicit model")
    return models[0]
