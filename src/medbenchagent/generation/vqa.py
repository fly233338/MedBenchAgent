"""Deterministic candidate validation and answer encoding for vqa tasks."""

from __future__ import annotations

from typing import Any


def supports(candidate: Any) -> bool:
    return (
        getattr(candidate, "expected_answer", None) is not None
        and getattr(candidate, "expected_evidence", None) is not None
    )


def _validate(candidate: Any, response: object) -> dict[str, Any]:
    if candidate.expected_answer is None or candidate.expected_evidence is None:
        raise ValueError("Mapped letter generation requires a fixed answer and evidence.")
    if not isinstance(response, dict) or set(response) != {"question"}:
        raise ValueError("Mapped letter generation must contain only question.")
    question = response["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    return {
        "question": question.strip(),
        "answer": candidate.expected_answer,
        "evidence": dict(candidate.expected_evidence),
    }


def encode_answer(candidate: Any, generated: dict[str, Any]) -> tuple[list[str], str]:
    options = list(candidate.task["options"])
    return options, chr(ord("A") + options.index(generated["answer"]))
