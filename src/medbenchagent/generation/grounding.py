"""Deterministic candidate validation and answer encoding for grounding tasks."""

from __future__ import annotations

from typing import Any
from medbenchagent.planning.protocol import resolve_answer


def _target_bbox(candidate: Any) -> list[float] | None:
    resolved = resolve_answer(candidate.record, candidate.task["protocol"])
    return resolved["answer"] if resolved else None


def _bbox_evidence(candidate: Any) -> list[dict] | None:
    resolved = resolve_answer(candidate.record, candidate.task["protocol"])
    return resolved["evidence"] if resolved else None


def _visual_target_text(candidate: Any) -> str | None:
    resolved = resolve_answer(candidate.record, candidate.task["protocol"])
    return resolved["target"] if resolved else None


def supports(candidate: Any) -> bool:
    return resolve_answer(candidate.record, candidate.task["protocol"]) is not None


def _validate(candidate: Any, response: object) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"question"}:
        raise ValueError("Grounding generation must contain only question.")
    question = response["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    answer = _target_bbox(candidate)
    if answer is None:
        raise ValueError("Grounding answer must have one unique annotation bbox.")
    evidence = _bbox_evidence(candidate)
    if evidence is None:
        raise ValueError("Grounding answer must have one exact source witness.")
    return {"question": question.strip(), "answer": answer, "evidence": evidence}


def encode_answer(_candidate: Any, generated: dict[str, Any]) -> tuple[None, list[float]]:
    return None, generated["answer"]
