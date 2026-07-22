"""Single-target bbox grounding generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

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
- Supporting annotation: {supporting_annotation}

Requirements:
1. Ask for the location of the annotated target; the question may name its finding category, but avoid following the canonical template verbatim and naturally vary wording, syntax, and word order.
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
- Supporting annotation: {supporting_annotation}
- Question: {question}
- Answer: {answer}

Approve only when:
1. The question identifies the same annotated target as the answer box.
2. The question does not reveal coordinates or clearly hint at the answer.
3. The answer exactly matches the supplied annotation box.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()


def _valid_bbox(value: object) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        return None
    bbox = [float(item) for item in value]
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


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
    boxes = _boxes(candidate.supporting_annotation)
    return boxes[0] if len(boxes) == 1 else None


def supports(candidate: Any) -> bool:
    return _target_bbox(candidate) is not None


def _task_definition(candidate: Any) -> str:
    return json.dumps(
        {"description": candidate.task["description"], "require": candidate.task["require"]},
        ensure_ascii=False,
    )


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    return GROUNDING_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=benchmark_info.get("modality", ""),
        task_id=candidate.task_id,
        task_definition=_task_definition(candidate),
        generation_logic=candidate.task["generation_logic"],
        canonical_question=candidate.task["question_templates"][0],
        supporting_annotation=json.dumps(candidate.supporting_annotation, ensure_ascii=False),
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
    return {"question": question.strip(), "answer": answer}


def generate(
    candidate: Any, benchmark_info: dict[str, object], question_temperature: float
) -> dict[str, Any]:
    prompt = _prompt(candidate, benchmark_info)
    try:
        response = complete_json(prompt, "", temperature=question_temperature)
        return _validate(candidate, response)
    except ValueError as first_error:
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
    response = complete_json(
        GROUNDING_REVIEW_PROMPT.format(
            task_definition=_task_definition(candidate),
            generation_logic=candidate.task["generation_logic"],
            supporting_annotation=json.dumps(candidate.supporting_annotation, ensure_ascii=False),
            question=generated["question"],
            answer=json.dumps(generated["answer"]),
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
