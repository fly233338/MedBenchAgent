"""Single-target bbox grounding generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

from medbenchagent.construction import event_operation
from medbenchagent.generation.context import (
    image_modality,
    locate_bbox_evidence,
    normalized_bbox,
    record_values,
)
from medbenchagent.llm import complete_json

GROUNDING_GENERATION_PROMPT = """
Generate one grounding question for a medical image benchmark.

Context:
- Language: {language}
- Modality: {modality}
- Task ID: {task_id}
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Canonical question: {canonical_question}
- Supporting data: {supporting_data}

Requirements:
1. {task_requirement}
2. Do not include coordinates, reports, annotations, labels, or unsupported details in the question.
3. Copy the exact annotated xyxy bounding box as the answer. Do not infer a new box.

Return JSON only:
{{"question": "question stem", "answer": [x1, y1, x2, y2]}}
""".strip()

GROUNDING_REVIEW_PROMPT = """
Review one grounding VQA item.

Context:
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Supporting data: {supporting_data}
- Question: {question}
- Answer: {answer}

Approve only when:
1. {review_requirement}
2. The question does not reveal coordinates or clearly hint at the answer.
3. The answer exactly matches the supplied annotation box.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()


def _valid_bbox(value: object) -> list[float] | None:
    return normalized_bbox(value)


def _boxes(value: object) -> list[list[float]]:
    bbox = _valid_bbox(value)
    if bbox is not None:
        return [bbox]
    if isinstance(value, dict):
        return [bbox for key, item in value.items() if key != "_source_index" for bbox in _boxes(item)]
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


def _task_definition(candidate: Any) -> str:
    return json.dumps(
        {"description": candidate.task["description"], "require": candidate.task["require"]},
        ensure_ascii=False,
    )


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    visual_text = _visual_target_text(candidate)
    if candidate.task_id == "visual_grounding":
        task_requirement = (
            f"Ask for the location of the explicitly named finding {visual_text!r}. Do not use a different "
            "finding name, and vary wording from the canonical template."
        )
    else:
        task_requirement = (
            "Ask only for the location of a lesion or abnormal region. Do not name, describe, or otherwise "
            "identify the finding category, and vary wording from the canonical template."
        )
    return GROUNDING_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=image_modality(candidate),
        task_id=candidate.task_id,
        task_definition=_task_definition(candidate),
        generation_logic=candidate.task["generation_logic"],
        canonical_question=candidate.task["question_templates"][0],
        supporting_data=json.dumps(candidate.supporting_record, ensure_ascii=False),
        task_requirement=task_requirement,
    )


def _validate(candidate: Any, response: object) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"question", "answer"}:
        raise ValueError("Grounding generation must contain only question and answer.")
    question = response["question"]
    answer = _valid_bbox(response["answer"])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    if answer is None or answer != _target_bbox(candidate):
        raise ValueError("Grounding answer must exactly match the unique annotation bbox.")
    evidence = _bbox_evidence(candidate)
    if evidence is None:
        raise ValueError("Grounding answer must have one exact source witness.")
    return {"question": question.strip(), "answer": answer, "evidence": evidence}


def generate(
    candidate: Any, benchmark_info: dict[str, object], question_temperature: float
) -> dict[str, Any]:
    prompt = _prompt(candidate, benchmark_info)
    try:
        response = complete_json(prompt, "", temperature=question_temperature)
        return _validate(candidate, response)
    except ValueError as first_error:
        with event_operation("repair"):
            repaired = complete_json(
                "Repair one grounding VQA response. Return JSON only.\n"
                f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
        return _validate(candidate, repaired)


def review(
    candidate: Any, generated: dict[str, Any], _benchmark_info: dict[str, object]
) -> tuple[bool, str]:
    review_requirement = (
        "The question identifies the same named finding as the answer box."
        if candidate.task_id == "visual_grounding"
        else "The question requests only a generic lesion or abnormal region and does not name a finding."
    )
    response = complete_json(
        GROUNDING_REVIEW_PROMPT.format(
            task_definition=_task_definition(candidate),
            generation_logic=candidate.task["generation_logic"],
            supporting_data=json.dumps(candidate.supporting_record, ensure_ascii=False),
            question=generated["question"],
            answer=json.dumps(generated["answer"]),
            review_requirement=review_requirement,
        ),
        "",
    )
    approved = (
        isinstance(response, dict)
        and set(response) == {"approved", "reason"}
        and response.get("approved") is True
        and isinstance(response.get("reason"), str)
    )
    reason = response.get("reason") if isinstance(response, dict) else "Invalid review response."
    return approved, str(reason)


def encode_answer(_candidate: Any, generated: dict[str, Any]) -> tuple[None, list[float]]:
    return None, generated["answer"]
