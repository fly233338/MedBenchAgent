"""Small OpenAI-compatible client for the configured model."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, TypeVar

from langsmith.wrappers import wrap_openai
from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from medbenchagent.runtime.construction import record_llm_call
from medbenchagent.settings import get_llm_api_key, get_settings

_CLIENT: OpenAI | None = None
_CLIENT_CONFIG: tuple[str, str, float, int, str] | None = None


class ModelUnavailable(RuntimeError):
    """Raised when the planning model is not configured or returns invalid JSON."""


class StructuredOutputError(ModelUnavailable):
    """Raised after model output cannot be repaired to satisfy a contract."""


T = TypeVar("T", bound=BaseModel)


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
    """Return a JSON object from the configured OpenAI-compatible endpoint."""
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
        raise ModelUnavailable(f"LLM request failed: {type(exc).__name__}.") from exc
    content = response.choices[0].message.content
    if not content:
        record_llm_call(status="failure", started=started, error_type="empty_response", response=response)
        raise ModelUnavailable("LLM returned an empty response.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        record_llm_call(status="failure", started=started, error_type="invalid_json", response=response)
        raise ModelUnavailable("LLM did not return valid JSON.") from exc
    if not isinstance(result, dict):
        record_llm_call(status="failure", started=started, error_type="invalid_json_object", response=response)
        raise ModelUnavailable("LLM JSON response must be an object.")
    record_llm_call(status="success", started=started, response=response)
    return result


def complete_validated_json(
    system_prompt: str, user_prompt: str, schema: type[T], *,
    validate: Callable[[T], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_retry: Callable[[], None] | None = None,
) -> T:
    """Repair model contract errors, independently of API transport retries."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    previous: tuple[str, str] | None = None
    for attempt in range(3):
        started = time.perf_counter()
        try:
            response = _client().chat.completions.create(
                model=get_settings().llm.model,
                temperature=get_settings().llm.default_temperature,
                response_format={"type": "json_object"},
                messages=messages,
                stream=on_chunk is not None,
            )
            if on_chunk is None:
                raw = response.choices[0].message.content or ""
            else:
                pieces = []
                for chunk in response:
                    if chunk.choices:
                        piece = chunk.choices[0].delta.content or ""
                        if piece:
                            pieces.append(piece)
                            on_chunk(piece)
                raw = "".join(pieces)
        except APIError as exc:
            record_llm_call(status="failure", started=started, error_type=type(exc).__name__)
            raise ModelUnavailable(f"LLM request failed: {type(exc).__name__}.") from exc
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            parsed = schema.model_validate(data)
            if validate is not None:
                validate(parsed)
        except (ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                errors = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                          for item in exc.errors()[:5]]
                feedback = "; ".join(errors)
            else:
                feedback = str(exc)
            feedback = feedback[:600]
            record_llm_call(status="failure", started=started, error_type="validation_error")
            if attempt == 2 or (raw, feedback) == previous:
                raise StructuredOutputError(f"LLM structured response failed validation: {feedback}") from exc
            previous = raw, feedback
            messages.extend([{"role": "assistant", "content": raw[:6000]},
                             {"role": "user", "content":
                              f"Correct your JSON. Contract error: {feedback}. Return only a corrected JSON object."}])
            if on_retry is not None:
                on_retry()
            continue
        record_llm_call(status="success", started=started,
                        response=response if on_chunk is None else None)
        return parsed
    raise AssertionError("unreachable")
