"""Small OpenAI-compatible client for the deployed DeepSeek model."""

from __future__ import annotations

import json
import time
from typing import Any

from langsmith.wrappers import wrap_openai
from openai import APIError, OpenAI

from medbenchagent.construction import record_llm_call
from medbenchagent.settings import get_llm_api_key, get_settings, load_project_env as _load_project_env

_CLIENT: OpenAI | None = None
_CLIENT_CONFIG: tuple[str, str, float, int, str] | None = None


class ModelUnavailable(RuntimeError):
    """Raised when the planning model is not configured or returns invalid JSON."""


def load_project_env() -> None:
    """Compatibility wrapper for the centralized environment loader."""
    _load_project_env()


def clear_client_cache() -> None:
    """Drop the process-local client; intended for configuration changes and tests."""
    global _CLIENT, _CLIENT_CONFIG
    if _CLIENT is not None:
        _CLIENT.close()
    _CLIENT = None
    _CLIENT_CONFIG = None


def _client() -> OpenAI:
    global _CLIENT, _CLIENT_CONFIG
    settings = get_settings().llm
    api_key = get_llm_api_key() or "local-vllm"
    config = (api_key, settings.base_url, settings.timeout_seconds, settings.max_retries, settings.trace_name)
    if _CLIENT is None or _CLIENT_CONFIG != config:
        if _CLIENT is not None:
            _CLIENT.close()
        raw_client = OpenAI(
            api_key=api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
        _CLIENT = wrap_openai(raw_client, chat_name=settings.trace_name)
        _CLIENT_CONFIG = config
    return _CLIENT


def complete_json(
    system_prompt: str, user_prompt: str, *, temperature: float | None = None
) -> dict[str, Any]:
    """Return a JSON object from the configured DeepSeek-compatible endpoint."""
    started = time.perf_counter()
    settings = get_settings().llm
    model = settings.model

    try:
        response = _client().chat.completions.create(
            model=model,
            temperature=settings.default_temperature if temperature is None else temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIError as exc:
        record_llm_call(status="failure", started=started, error_type=type(exc).__name__)
        raise ModelUnavailable(f"DeepSeek request failed: {type(exc).__name__}.") from exc
    content = response.choices[0].message.content
    if not content:
        record_llm_call(status="failure", started=started, error_type="empty_response", response=response)
        raise ModelUnavailable("DeepSeek returned an empty response.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        record_llm_call(status="failure", started=started, error_type="invalid_json", response=response)
        raise ModelUnavailable("DeepSeek did not return valid JSON.") from exc
    if not isinstance(result, dict):
        record_llm_call(status="failure", started=started, error_type="invalid_json_object", response=response)
        raise ModelUnavailable("DeepSeek JSON response must be an object.")
    record_llm_call(status="success", started=started, response=response)
    return result
