"""Small OpenAI-compatible client for the configured model."""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any, Callable, TypeVar

from langsmith.wrappers import wrap_openai
from openai import APIError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from medbenchagent.runtime.construction import record_llm_call, record_llm_validation_error
from medbenchagent.runtime.model_services import resolve_model_service
from medbenchagent.settings import get_llm_api_key, get_settings

_CLIENTS: dict[tuple, OpenAI] = {}
_CLIENT_LOCK = Lock()
_VALIDATION_OUTPUT_LIMIT = 2400


def _validation_output_excerpt(value: str) -> str:
    """Keep validation events useful without storing full model responses."""
    if len(value) <= _VALIDATION_OUTPUT_LIMIT:
        return value
    head = 1600
    tail = _VALIDATION_OUTPUT_LIMIT - head
    omitted = len(value) - head - tail
    return f"{value[:head]}\n...[{omitted} chars truncated]...\n{value[-tail:]}"


class ModelUnavailable(RuntimeError):
    """Raised when the model service cannot complete a request."""


class StructuredOutputError(ValueError):
    """Raised after model output cannot be repaired to satisfy a contract."""


class RequestContextBudgetError(ValueError):
    """The complete serialized request cannot fit without losing required context."""

    def __init__(self, actual: int, limit: int):
        super().__init__(f"Model request requires {actual} characters, exceeding planning.context_chars={limit}; "
                         "narrow the evidence query or increase the context budget. Required context was not truncated.")
        self.diagnostics = {"request_chars": actual, "context_chars": limit}


T = TypeVar("T", bound=BaseModel)


def clear_client_cache() -> None:
    """Close cached clients only at shutdown or in tests, after requests have stopped."""
    with _CLIENT_LOCK:
        for client in _CLIENTS.values():
            client.close()
        _CLIENTS.clear()


def _client() -> OpenAI:
    settings = get_settings().llm
    api_key = get_llm_api_key() or "local-vllm"
    config = (api_key, settings.base_url, settings.timeout_seconds, settings.max_retries)
    with _CLIENT_LOCK:
        if config not in _CLIENTS:
            raw_client = OpenAI(
                api_key=api_key,
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=settings.max_retries,
            )
            _CLIENTS[config] = wrap_openai(raw_client, chat_name="medbench-llm")
        return _CLIENTS[config]


class JsonObject(BaseModel):
    model_config = ConfigDict(extra="allow")


def complete_json(
    system_prompt: str, user_prompt: str, schema: type[BaseModel] = JsonObject,
    validate: Callable[[BaseModel], None] | None = None, *, temperature: float | None = None,
    request_char_budget: int | None = None,
) -> dict[str, Any]:
    """Return validated JSON using the same bounded repair path as agent decisions."""
    return complete_validated_json(system_prompt, user_prompt, schema,
                                   temperature=temperature, validate=validate,
                                   request_char_budget=request_char_budget).model_dump(exclude_unset=True)


def complete_validated_json(
    system_prompt: str, user_prompt: str, schema: type[T], *,
    validate: Callable[[T], None] | None = None,
    temperature: float | None = None,
    on_chunk: Callable[[str], None] | None = None,
    on_retry: Callable[[], None] | None = None,
    request_char_budget: int | None = None,
) -> T:
    """Repair model contract errors, independently of API transport retries."""
    settings = get_settings()
    model = (settings.llm.model if getattr(settings, "service_status", {}).get("chat") == "available" else
             resolve_model_service("Chat", settings.llm.base_url, settings.llm.model, get_llm_api_key()))
    messages = [{"role": "system", "content": system_prompt + "\nReturn JSON matching this schema:\n"
                 + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
                {"role": "user", "content": user_prompt}]
    previous: tuple[str, str] | None = None
    for attempt in range(3):
        started = time.perf_counter()
        try:
            request = {
                "model": model,
                "temperature": get_settings().llm.default_temperature if temperature is None else temperature,
                "response_format": {"type": "json_object"},
                "messages": messages,
                "stream": on_chunk is not None,
            }
            if settings.llm.reasoning_effort:
                request["reasoning_effort"] = settings.llm.reasoning_effort
            if request_char_budget is not None:
                actual = len(json.dumps(request, ensure_ascii=False))
                if actual > request_char_budget:
                    raise RequestContextBudgetError(actual, request_char_budget)
            response = _client().chat.completions.create(**request)
            if on_chunk is None:
                raw = (response.choices[0].message.content or "") if response.choices else ""
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
            if isinstance(exc, APITimeoutError):
                raise ModelUnavailable("LLM request timed out after the configured limit; no immediate retry was attempted.") from exc
            raise ModelUnavailable(f"LLM request failed: {type(exc).__name__}.") from exc
        phase = "json_decode_error"
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            phase = "tool_action_validation_error" if schema.__name__ == "ToolAction" else "schema_validation_error"
            parsed = schema.model_validate(data)
            if validate is not None:
                phase = "business_validation_error"
                validate(parsed)
        except (ValueError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                errors = [f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                          for item in exc.errors()]
                feedback = "; ".join(errors)
            else:
                errors = [str(exc)]
                feedback = str(exc)
            diagnostic = [f"phase={phase}", f"schema={schema.__name__}", *errors]
            repair_guidance = (
                " Correct only the reported contract errors, preserve all already valid fields, "
                "and return one complete JSON object."
            )
            feedback += repair_guidance
            record_llm_validation_error(
                errors=diagnostic,
                model_output=_validation_output_excerpt(str(raw)),
            )
            record_llm_call(status="failure", started=started, error_type=phase)
            if attempt == 2 or (raw, feedback) == previous:
                raise StructuredOutputError(f"LLM structured response failed validation: {feedback}") from exc
            previous = raw, feedback
            messages.extend([{"role": "assistant", "content": raw},
                             {"role": "user", "content":
                              f"Correct your JSON. Contract error: {feedback} Return only a corrected JSON object."}])
            if on_retry is not None:
                on_retry()
            continue
        record_llm_call(status="success", started=started,
                        response=response if on_chunk is None else None)
        return parsed
    raise AssertionError("unreachable")
