"""Deterministic candidate validation and answer encoding for grounding tasks."""

from __future__ import annotations

from typing import Any

from medbenchagent.generation.context import (
    locate_bbox_evidence,
    normalized_bbox,
    record_values,
)


def _valid_bbox(value: object) -> list[float] | None:
    return normalized_bbox(value)


def _boxes(value: object) -> list[list[float]]:
    bbox = _valid_bbox(value)
    if bbox is not None:
        return [bbox]
    if isinstance(value, dict):
        return [
            bbox for key, item in value.items() if key != "_source_index" for bbox in _boxes(item)
        ]
    if isinstance(value, list):
        return [bbox for item in value for bbox in _boxes(item)]
    return []


def _target_bbox(candidate: Any) -> list[float] | None:
    boxes = _boxes(candidate.supporting_record)
    return boxes[0] if len(boxes) == 1 else None


def _bbox_evidence(candidate: Any) -> list[dict[str, object]] | None:
    target = _target_bbox(candidate)
    if target is None:
        return None
    try:
        return locate_bbox_evidence(
            record_values(candidate),
            candidate.task.get("supporting_fields"),
            target,
        )
    except ValueError:
        return None


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [
            text
            for key, item in value.items()
            if key not in {"_source_index", "bbox"}
            for text in _strings(item)
        ]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def _named_region_text(value: object, target: list[float]) -> str | None:
    if isinstance(value, dict):
        if _boxes(value) == [target] and (texts := _strings(value)):
            return texts[0]
        for item in value.values():
            if text := _named_region_text(item, target):
                return text
    elif isinstance(value, list):
        for item in value:
            if text := _named_region_text(item, target):
                return text
    return None


def _visual_target_text(candidate: Any) -> str | None:
    target = _target_bbox(candidate)
    if target is None:
        return None
    return _named_region_text(candidate.supporting_record, target)


def supports(candidate: Any) -> bool:
    if _bbox_evidence(candidate) is None:
        return False
    return candidate.task_id != "visual_grounding" or _visual_target_text(candidate) is not None


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
