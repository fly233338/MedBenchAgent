"""Letter-choice VQA generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

from medbenchagent.generation.context import annotation_modality
from medbenchagent.construction import event_operation
from medbenchagent.llm import complete_json

VQA_GENERATION_PROMPT = """
Generate one letter-choice question for a medical image VQA benchmark.
Context:
- Language: {language}
- Modality: {modality}
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Canonical question: {canonical_question}
- Candidate options: {candidate_options}
- Supporting annotation: {supporting_annotation}
Requirements:
1. Generate a controlled linguistic variation of the question anchor while preserving the
   exact medical meaning and task intent. Do not modify, generalize, replace, remove, or
   reinterpret medical concepts, including anatomical structures, organs, tissues,
   imaging modalities, pathology terms, clinical attributes, and task-specific concepts.
   Do not change the capability being evaluated or introduce new interpretations.
2. Apply only surface-level linguistic changes to create variation. You may adjust sentence
   structure, word order, question formulation, function words, prepositions, or general
   phrasing. The generated question should be linguistically different but semantically
   equivalent to the question anchor. Keep modality-specific and medical terminology
   explicit when they appear in the anchor.
3. The question must be answerable from the image and candidate options only.
   Do not mention reports, annotations, labels, metadata, or unsupported clinical details.
4. Do not reveal, describe, or hint at the correct answer in the question.
5. Normalize the annotation-supported answer to exactly one candidate option.
   Do not infer beyond the supporting annotation.
Return JSON only:
{{"question": "question stem", "answer": "one exact candidate option"}}
""".strip()

VQA_REVIEW_PROMPT = """
Review one generated letter-choice VQA item.

Context:
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Candidate options: {candidate_options}
- Supporting annotation: {supporting_annotation}
- Question: {question}
- Answer: {answer}

Approve only when:
1. The question does not reveal or clearly hint at the answer.
2. The answer is exactly supported by the annotation.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()


def supports(candidate: Any) -> bool:
    return True


def _task_definition(candidate: Any) -> str:
    return json.dumps(
        {"description": candidate.task["description"], "require": candidate.task["require"]},
        ensure_ascii=False,
    )


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    return VQA_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=annotation_modality(candidate),
        task_id=candidate.task_id,
        task_definition=_task_definition(candidate),
        generation_logic=candidate.task["generation_logic"],
        canonical_question=candidate.task["question_templates"][0],
        candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
        supporting_annotation=json.dumps(candidate.supporting_annotation, ensure_ascii=False),
    )


def _validate(candidate: Any, response: object) -> dict[str, str]:
    if not isinstance(response, dict) or set(response) != {"question", "answer"}:
        raise ValueError("Letter generation must contain only question and answer.")
    question = response["question"]
    answer = response["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    if not isinstance(answer, str) or answer not in candidate.task["options"]:
        raise ValueError("Letter answer must be one exact option.")
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
                "Repair one letter-choice VQA response. Return JSON only.\n"
                f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
        return _validate(candidate, repaired)


def review(
    candidate: Any, generated: dict[str, str], _benchmark_info: dict[str, object]
) -> tuple[bool, str]:
    response = complete_json(
        VQA_REVIEW_PROMPT.format(
            task_definition=_task_definition(candidate),
            generation_logic=candidate.task["generation_logic"],
            candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
            supporting_annotation=json.dumps(candidate.supporting_annotation, ensure_ascii=False),
            question=generated["question"],
            answer=generated["answer"],
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


def encode_answer(candidate: Any, generated: dict[str, str]) -> tuple[list[str], str]:
    options = list(candidate.task["options"])
    return options, chr(ord("A") + options.index(generated["answer"]))
