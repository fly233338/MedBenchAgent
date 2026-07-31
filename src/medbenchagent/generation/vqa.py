"""Letter-choice VQA generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

from medbenchagent.construction import event_operation
from medbenchagent.generation.context import (
    image_modality,
    record_values,
    validate_evidence,
)
from medbenchagent.llm import complete_json

UNSUPPORTED_DIAGNOSIS_VALUES = frozenset(
    {
        "unknown",
        "not available",
        "not applicable",
        "n/a",
        "none",
        "not specified",
        "unspecified",
    }
)

LEGACY_VQA_GENERATION_PROMPT = """
Generate one letter-choice question for a medical image VQA benchmark.
Context:
- Language: {language}
- Modality: {modality}
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Selected question template: {selected_question_template}
- Candidate options: {candidate_options}
- Supporting fields: {supporting_fields}
- Supporting data: {supporting_data}
Requirements:
1. Generate a controlled linguistic variation of the selected question template while preserving the
   exact medical meaning and task intent. Do not modify, generalize, replace, remove, or
   reinterpret medical concepts, including anatomical structures, organs, tissues,
   imaging modalities, pathology terms, clinical attributes, and task-specific concepts.
   Do not change the capability being evaluated or introduce new interpretations.
2. Apply only surface-level linguistic changes to create variation. You may adjust sentence
   structure, word order, question formulation, function words, prepositions, or general
   phrasing. The generated question should be linguistically different but semantically
   equivalent to the selected template; use a distinctly different grammatical structure when possible.
   Avoid stock openers such as Based on…. Keep modality-specific and medical terminology
   explicit when they appear in the selected template.
3. The question must be answerable from the image and candidate options only.
   Do not mention reports, annotations, labels, metadata, or unsupported clinical details.
4. Do not reveal, describe, or hint at the correct answer in the question.
5. Normalize the data-supported answer to exactly one candidate option.
   Do not infer beyond the supporting data.
6. Copy exactly one raw scalar value that supports the answer. Use one exact
   supporting-field path. For every [*] in that path, provide its original source-array
   index in order. Use `_source_index` for projected object-array entries and zero-based
   positions for scalar arrays. Copy the value exactly without changing case, spelling, or type.
Return JSON only:
{{"question": "question stem", "answer": "one exact candidate option",
  "evidence": {{"path": "one exact supporting field", "indices": [0], "value": "exact raw value"}}}}
""".strip()

VQA_GENERATION_PROMPT = """
Generate only the question stem for one letter-choice medical-image benchmark item.

Fixed context:
- Language: {language}
- Modality: {modality}
- Task definition: {task_definition}
- Question type: {question_type}
- Answer field: {answer_field}
- Fixed answer option: {expected_answer}
- Fixed raw evidence: {expected_evidence}
- Selected template: {selected_question_template}
- Options: {candidate_options}

Requirements:
1. Preserve the declared question type and answer axis.
2. Apply surface-level linguistic variation only, using a distinctly different grammatical structure when possible.
   Avoid stock openers such as Based on….
3. Do not switch property, target, scope, or granularity.
4. Do not reveal or hint at the answer.
5. Do not mention annotations, labels, metadata, or reports.
6. Do not choose or reinterpret the answer or evidence.

Return JSON only:
{{"question": "question stem"}}
""".strip()

LEGACY_VQA_REVIEW_PROMPT = """
Review one generated letter-choice VQA item.

Context:
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Candidate options: {candidate_options}
- Supporting data: {supporting_data}
- Question: {question}
- Answer: {answer}
- Answer evidence: {evidence}

Approve only when:
1. The question does not reveal or clearly hint at the answer.
2. The raw evidence value reasonably maps to the selected answer option.
3. The question asks for the concept represented by that evidence and is consistent
   with the supporting data.
4. When several findings or values are present, the question and selected evidence do
   not create an obviously arbitrary or ambiguous target.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()

VQA_REVIEW_PROMPT = """
Review one generated letter-choice VQA item whose answer and evidence were fixed by code.

Context:
- Task definition: {task_definition}
- Question type: {question_type}
- Candidate options: {candidate_options}
- Supporting data: {supporting_data}
- Question: {question}
- Fixed answer: {answer}

Approve only when:
1. The question asks exactly for the declared question type and answer axis.
2. It is answerable using the image and supplied options.
3. It does not reveal or strongly hint at the fixed answer.
4. It does not switch property, target, scope, granularity, or clinical concept.
5. The options are mutually exclusive for the question as written.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()


def _scalar_values(value: object) -> list[object]:
    if isinstance(value, dict):
        return [
            scalar
            for key, item in value.items()
            if key != "_source_index"
            for scalar in _scalar_values(item)
        ]
    if isinstance(value, list):
        return [scalar for item in value for scalar in _scalar_values(item)]
    return [] if value is None or value == "" else [value]


def supports(candidate: Any) -> bool:
    if getattr(candidate, "expected_answer", None) is not None:
        return True
    if candidate.task_id != "diagnosis":
        return True
    values = _scalar_values(candidate.supporting_record)
    if len(values) != 1 or not isinstance(values[0], str):
        return False
    diagnosis = values[0].strip()
    return bool(
        diagnosis
        and "&" not in diagnosis
        and diagnosis.casefold() not in UNSUPPORTED_DIAGNOSIS_VALUES
    )


def _task_definition(candidate: Any) -> str:
    return json.dumps(
        {
            "description": candidate.task["description"],
            "require": candidate.task["require"],
            "question_type_id": candidate.task.get("question_type_id"),
        },
        ensure_ascii=False,
    )


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    if (
        getattr(candidate, "expected_answer", None) is not None
        and getattr(candidate, "expected_evidence", None) is not None
    ):
        return VQA_GENERATION_PROMPT.format(
            language=benchmark_info.get("language", ""),
            modality=image_modality(candidate),
            task_definition=_task_definition(candidate),
            question_type=candidate.task.get("question_type_id", candidate.task_id),
            answer_field=candidate.task["answer_field"],
            expected_answer=candidate.expected_answer,
            expected_evidence=json.dumps(candidate.expected_evidence, ensure_ascii=False),
            selected_question_template=candidate.question_template,
            candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
        )
    return LEGACY_VQA_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=image_modality(candidate),
        task_id=candidate.task_id,
        task_definition=_task_definition(candidate),
        generation_logic=candidate.task["generation_logic"],
        selected_question_template=candidate.question_template,
        candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
        supporting_fields=json.dumps(candidate.task["supporting_fields"], ensure_ascii=False),
        supporting_data=json.dumps(candidate.supporting_record, ensure_ascii=False),
    )


def _validate(candidate: Any, response: object) -> dict[str, Any]:
    if (
        getattr(candidate, "expected_answer", None) is not None
        and getattr(candidate, "expected_evidence", None) is not None
    ):
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
    if not isinstance(response, dict) or set(response) != {"question", "answer", "evidence"}:
        raise ValueError("Letter generation must contain only question, answer, and evidence.")
    question = response["question"]
    answer = response["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    if not isinstance(answer, str) or answer not in candidate.task["options"]:
        raise ValueError("Letter answer must be one exact option.")
    evidence = validate_evidence(
        record_values(candidate),
        candidate.task.get("supporting_fields"),
        response["evidence"],
    )
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
                "Repair one letter-choice VQA response. Return JSON only.\n"
                f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
        return _validate(candidate, repaired)


def review(
    candidate: Any, generated: dict[str, Any], _benchmark_info: dict[str, object]
) -> tuple[bool, str]:
    if getattr(candidate, "expected_answer", None) is not None:
        prompt = VQA_REVIEW_PROMPT.format(
            task_definition=_task_definition(candidate),
            question_type=candidate.task.get("question_type_id", candidate.task_id),
            candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
            supporting_data=json.dumps(candidate.supporting_record, ensure_ascii=False),
            question=generated["question"],
            answer=generated["answer"],
        )
    else:
        prompt = LEGACY_VQA_REVIEW_PROMPT.format(
            task_definition=_task_definition(candidate),
            generation_logic=candidate.task["generation_logic"],
            candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
            supporting_data=json.dumps(candidate.supporting_record, ensure_ascii=False),
            question=generated["question"],
            answer=generated["answer"],
            evidence=json.dumps(generated["evidence"], ensure_ascii=False),
        )
    response = complete_json(
        prompt,
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


def encode_answer(candidate: Any, generated: dict[str, Any]) -> tuple[list[str], str]:
    options = list(candidate.task["options"])
    return options, chr(ord("A") + options.index(generated["answer"]))
