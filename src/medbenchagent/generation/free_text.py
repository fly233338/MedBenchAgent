"""Direct-reference free-text generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

from medbenchagent.construction import event_operation
from medbenchagent.generation.context import (
    image_modality,
    locate_scalar_evidence,
    record_values,
)
from medbenchagent.llm import complete_json

FREE_TEXT_GENERATION_PROMPT = """
Generate one free-text medical image benchmark item.

Context:
- Language: {language}
- Modality: {modality}
- Task ID: {task_id}
- Medical task description: {medical_task_description}
- Generation logic: {generation_logic}
- Selected question template: {selected_question_template}
- Reference text: {reference_text}

Requirements:
1. Write one substantial semantic paraphrase of the selected question template; avoid following it verbatim and naturally vary wording, syntax, and word order.
2. Do not reveal or clearly hint at the reference text in the question.
3. The exact reference answer is fixed by code; do not generate or reinterpret an answer.

Return JSON only:
{{"question": "question stem"}}
""".strip()

FREE_TEXT_REVIEW_PROMPT = """
Review one direct-reference free-text VQA item.

Context:
- Medical task description: {medical_task_description}
- Generation logic: {generation_logic}
- Reference text: {reference_text}
- Question: {question}
- Answer: {answer}

Approve only when:
1. The question follows the medical task description and generation logic and does not reveal the answer.
2. The answer exactly equals the reference text.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()


def _texts(value: object) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() if key != "_source_index" for text in _texts(item)]
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


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    return FREE_TEXT_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=image_modality(candidate),
        task_id=candidate.task_id,
        medical_task_description=candidate.task["medical_task_description"],
        generation_logic=candidate.task["generation_logic"],
        selected_question_template=candidate.question_template,
        reference_text=json.dumps(_reference_text(candidate), ensure_ascii=False),
    )


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
                "Repair one free-text VQA question response. Return JSON only.\n"
                f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
        return _validate(candidate, repaired)


def review(
    candidate: Any, generated: dict[str, Any], _benchmark_info: dict[str, object]
) -> tuple[bool, str]:
    reference = _reference_text(candidate)
    response = complete_json(
        FREE_TEXT_REVIEW_PROMPT.format(
            medical_task_description=candidate.task["medical_task_description"],
            generation_logic=candidate.task["generation_logic"],
            reference_text=json.dumps(reference, ensure_ascii=False),
            question=generated["question"],
            answer=json.dumps(generated["answer"], ensure_ascii=False),
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


def encode_answer(_candidate: Any, generated: dict[str, Any]) -> tuple[None, str]:
    return None, generated["answer"]
