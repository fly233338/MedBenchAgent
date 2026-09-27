"""Shared test adapters for validated Harness decisions."""

import pytest
from dataclasses import replace
from pydantic import ValidationError

from medbenchagent.runtime.llm import StructuredOutputError


@pytest.fixture
def harness_model(monkeypatch):
    # Graph integration tests use scripted decisions and no live model endpoint.
    def resolved(settings):
        return replace(settings, llm=replace(settings.llm, model=settings.llm.model or "test-chat"),
                       service_status={"chat": "available", "embedding": "unavailable",
                                       "reranker": "unavailable"})

    monkeypatch.setattr("medbenchagent.runtime.model_services.resolve_run_services", resolved)
    monkeypatch.setattr("medbenchagent.runtime.model_services.verify_frozen_services", lambda _settings: None)
    def install(responder):
        def complete(system_prompt, user_prompt, schema, *, validate=None, on_retry=None):
            for attempt in range(3):
                raw = responder(system_prompt, user_prompt)
                try:
                    action = schema.model_validate(raw)
                    if validate is not None:
                        validate(action)
                except (ValueError, ValidationError) as exc:
                    complete.validation_errors.append(str(exc))
                    if attempt == 2:
                        raise StructuredOutputError(f"LLM structured response failed validation: {exc}") from exc
                    if on_retry is not None:
                        on_retry()
                    continue
                return action
            raise AssertionError("unreachable")

        complete.validation_errors = []
        monkeypatch.setattr("medbenchagent.runtime.harness.complete_validated_json", complete)
        return complete

    return install
