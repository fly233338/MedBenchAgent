"""Direct-reference free-text generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

from medbenchagent.generation.context import annotation_modality
from medbenchagent.construction import event_operation
from medbenchagent.llm import complete_json

FREE_TEXT_GENERATION_PROMPT = """
Generate one free-text medical image benchmark item.

Context:
- Language: {language}
- Modality: {modality}
- Task ID: {task_id}
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Canonical question: {canonical_question}
- Reference text: {reference_text}

Requirements:
1. Write one substantial semantic paraphrase of the canonical question; avoid following its template verbatim and naturally vary wording, syntax, and word order.
2. Do not reveal or clearly hint at the reference text in the question.
3. Copy the reference text exactly as the answer. Do not summarize, rewrite, or add medical inference.

Return JSON only:
{{"question": "question stem", "answer": "exact reference text"}}
""".strip()

FREE_TEXT_REVIEW_PROMPT = """
Review one direct-reference free-text VQA item.

Context:
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Reference text: {reference_text}
- Question: {question}
- Answer: {answer}

Approve only when:
1. The question follows the task definition and does not reveal the answer.
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
    texts = _texts(candidate.supporting_annotation)
    return texts[0] if len(texts) == 1 else None


def supports(candidate: Any) -> bool:
    return _reference_text(candidate) is not None


def _task_definition(candidate: Any) -> str:
    return json.dumps(
        {"description": candidate.task["description"], "require": candidate.task["require"]},
        ensure_ascii=False,
    )


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    return FREE_TEXT_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=annotation_modality(candidate),
        task_id=candidate.task_id,
        task_definition=_task_definition(candidate),
        generation_logic=candidate.task["generation_logic"],
        canonical_question=candidate.task["question_templates"][0],
        reference_text=json.dumps(_reference_text(candidate), ensure_ascii=False),
    )


def _validate(candidate: Any, response: object) -> dict[str, str]:
    if not isinstance(response, dict) or set(response) != {"question", "answer"}:
        raise ValueError("Free-text generation must contain only question and answer.")
    question = response["question"]
    answer = response["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    if not isinstance(answer, str) or answer != _reference_text(candidate):
        raise ValueError("Free-text answer must exactly match the unique reference text.")
    return {"question": question.strip(), "answer": answer}


def generate(
    candidate: Any, benchmark_info: dict[str, object], question_temperature: float
) -> dict[str, str]:
    prompt = _prompt(candidate, benchmark_info)
    try:
        response = complete_json(prompt, "", temperature=question_temperature)
        return _validate(candidate, response)
    except ValueError as first_error:
        with event_operation("repair"):
            repaired = complete_json(
                "Repair one free-text VQA response. Return JSON only.\n"
                f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
        return _validate(candidate, repaired)


def review(
    candidate: Any, generated: dict[str, str], _benchmark_info: dict[str, object]
) -> tuple[bool, str]:
    reference = _reference_text(candidate)
    response = complete_json(
        FREE_TEXT_REVIEW_PROMPT.format(
            task_definition=_task_definition(candidate),
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


def encode_answer(_candidate: Any, generated: dict[str, str]) -> tuple[None, str]:
    return None, generated["answer"]
