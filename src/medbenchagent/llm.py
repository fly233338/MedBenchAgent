"""Small OpenAI-compatible client for the deployed DeepSeek model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import APIError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CLIENT: OpenAI | None = None
_CLIENT_CONFIG: tuple[str, str | None, float] | None = None


class ModelUnavailable(RuntimeError):
    """Raised when the planning model is not configured or returns invalid JSON."""


def load_project_env() -> None:
    """Load the repository's optional .env without overriding process-level settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


load_project_env()


def clear_client_cache() -> None:
    """Drop the process-local client; intended for configuration changes and tests."""
    global _CLIENT, _CLIENT_CONFIG
    if _CLIENT is not None:
        _CLIENT.close()
    _CLIENT = None
    _CLIENT_CONFIG = None


def _client() -> OpenAI:
    global _CLIENT, _CLIENT_CONFIG
    api_key = os.getenv("DEEPSEEK_API_KEY") or "local-vllm"
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    raw_timeout = os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ModelUnavailable("DEEPSEEK_TIMEOUT_SECONDS must be a positive number.") from exc
    if timeout <= 0:
        raise ModelUnavailable("DEEPSEEK_TIMEOUT_SECONDS must be a positive number.")
    config = (api_key, base_url, timeout)
    if _CLIENT is None or _CLIENT_CONFIG != config:
        if _CLIENT is not None:
            _CLIENT.close()
        raw_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        _CLIENT = wrap_openai(raw_client, chat_name="DeepSeek")
        _CLIENT_CONFIG = config
    return _CLIENT


def complete_json(system_prompt: str, user_prompt: str, *, temperature: float = 0) -> dict[str, Any]:
    """Return a JSON object from the configured DeepSeek-compatible endpoint."""
    model = os.getenv("DEEPSEEK_MODEL")
    if not model:
        raise ModelUnavailable("DEEPSEEK_MODEL must be configured.")

    try:
        response = _client().chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIError as exc:
        raise ModelUnavailable(f"DeepSeek request failed: {type(exc).__name__}.") from exc
    content = response.choices[0].message.content
    if not content:
        raise ModelUnavailable("DeepSeek returned an empty response.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ModelUnavailable("DeepSeek did not return valid JSON.") from exc
    if not isinstance(result, dict):
        raise ModelUnavailable("DeepSeek JSON response must be an object.")
    return result
