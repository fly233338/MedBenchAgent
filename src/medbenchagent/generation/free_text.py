"""Deterministic candidate validation and answer encoding for free_text tasks."""

from __future__ import annotations

from typing import Any

from medbenchagent.generation.context import (
    locate_scalar_evidence,
    record_values,
)


def _texts(value: object) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, dict):
        return [
            text for key, item in value.items() if key != "_source_index" for text in _texts(item)
        ]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    return []


def _reference_text(candidate: Any) -> str | None:
    texts = _texts(candidate.supporting_record)
    return texts[0] if len(texts) == 1 else None


def _reference_evidence(candidate: Any) -> dict[str, object] | None:
    reference = _reference_text(candidate)
    if reference is None:
        return None
    try:
        return locate_scalar_evidence(
            record_values(candidate),
            candidate.task.get("supporting_fields"),
            reference,
        )
    except ValueError:
        return None


def supports(candidate: Any) -> bool:
    return _reference_evidence(candidate) is not None


def _validate(candidate: Any, response: object) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"question"}:
        raise ValueError("Free-text generation must contain only question.")
    question = response["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    answer = _reference_text(candidate)
    if answer is None:
        raise ValueError("Free-text answer must have one unique reference text.")
    evidence = _reference_evidence(candidate)
    if evidence is None:
        raise ValueError("Free-text answer must have one exact source witness.")
    return {"question": question.strip(), "answer": answer, "evidence": evidence}


def encode_answer(_candidate: Any, generated: dict[str, Any]) -> tuple[None, str]:
    return None, generated["answer"]
