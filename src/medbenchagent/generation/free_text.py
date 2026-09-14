"""Deterministic candidate validation and answer encoding for free_text tasks."""

from __future__ import annotations

from typing import Any
from medbenchagent.planning.protocol import resolve_answer


def _reference_text(candidate: Any) -> str | None:
    resolved = resolve_answer(candidate.record, candidate.task["protocol"])
    return resolved["answer"] if resolved else None


def _reference_evidence(candidate: Any) -> dict | None:
    resolved = resolve_answer(candidate.record, candidate.task["protocol"])
    return resolved["evidence"] if resolved else None


def supports(candidate: Any) -> bool:
    return resolve_answer(candidate.record, candidate.task["protocol"]) is not None


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
