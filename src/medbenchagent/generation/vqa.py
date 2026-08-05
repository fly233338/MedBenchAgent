"""Letter-choice VQA generation, validation, and review."""

from __future__ import annotations

import json
from typing import Any

from medbenchagent.construction import event_operation
from medbenchagent.generation.context import image_modality
from medbenchagent.llm import complete_json

VQA_GENERATION_PROMPT = """
Generate only the question stem for one letter-choice medical-image benchmark item.

Fixed context:
- Language: {language}
- Modality: {modality}
- Medical task description: {medical_task_description}
- Generation logic: {generation_logic}
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

VQA_REVIEW_PROMPT = """
Review one generated letter-choice VQA item whose answer and evidence were fixed by code.

Context:

- Medical task description: {medical_task_description}
- Generation logic: {generation_logic}
- Answer field: {answer_field}
- Answer mapping: {answer_mapping}
- Candidate options: {candidate_options}
- Supporting data: {supporting_data}
- Selected question template: {selected_question_template}
- Question: {question}
- Fixed answer option: {answer}
- Fixed answer evidence: {evidence}

The fixed answer and evidence were deterministically derived by code from the approved
answer field and answer mapping. Treat them as ground truth for this review.
Do not replace, reinterpret, or select a different answer or evidence.

Decision policy:
Approve the item only when all required conditions below are satisfied.
Reject the item when any condition clearly and materially fails.

Do not reject an item for ordinary paraphrasing, stylistic variation, repeated medical
terminology, or minor awkwardness that does not change the task meaning or answerability.

Check the following:

1. Answer leakage

Approve when the question does not reveal or strongly hint at the fixed answer option
or its raw evidence value.

Reject when the question directly states the answer, repeats the answer-specific
annotation value, or includes a clue that makes the answer obvious without evaluating
the image.

2. Evidence-answer consistency

Approve when the fixed raw evidence belongs to the declared answer field and maps to
the fixed answer option according to the supplied answer mapping.

Reject when the question is inconsistent with the fixed answer or evidence, or when
the fixed answer and evidence cannot be reconciled with the declared mapping.

Do not change the fixed answer or evidence during review.

3. Attribute and concept preservation

Approve when the question asks for the same medical or visual concept, question type,
and answer axis defined by the task and represented by the fixed answer evidence.

Reject when the question replaces the intended concept with a related but different
attribute, finding, diagnosis, anatomical property, or clinical judgment.

4. Target preservation

Approve when the question refers to the same target finding, lesion, anatomical
structure, region, image, or case represented by the supporting data and fixed evidence.

Reject when the question changes the target or leaves the target clearly arbitrary when
multiple possible targets are present.

5. Scope and granularity preservation

Approve when the question preserves the intended scope and level of detail.

Reject when it changes between local and global assessment, single and multiple
findings, image and patient level, presence and subtype, or coarse and fine-grained
classification.

6. Question-answerability and option compatibility

Approve when the question is answerable from the image, medical task description, and candidate
options, and every candidate option is a natural and semantically valid possible answer
to the question as written.

Reject when the question depends on annotations, metadata, reports, unsupported facts,
or unseen clinical information, or when the grammatical form or response axis is
incompatible with the options.

7. Unsupported semantic additions

Approve when the question introduces no additional clinical meaning beyond the medical
task description, generation logic, declared answer field and mapping, selected template,
supporting data, and fixed evidence.

Reject when it adds unsupported confirmation status, pathology, causality, severity,
prognosis, treatment, temporal information, or other clinical assumptions.

8. Ambiguity

Approve when the question has one clear intended interpretation and one
evidence-supported answer.

Reject when the wording creates multiple plausible interpretations or when more than
one option could reasonably answer the question.

Return JSON only:
{{"approved": true, "reason": "short, specific reason"}}
or
{{"approved": false, "reason": "short, specific reason"}}
""".strip()


def supports(candidate: Any) -> bool:
    return (
        getattr(candidate, "expected_answer", None) is not None
        and getattr(candidate, "expected_evidence", None) is not None
    )


def _prompt(candidate: Any, benchmark_info: dict[str, object]) -> str:
    return VQA_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=image_modality(candidate),
        medical_task_description=candidate.task["medical_task_description"],
        generation_logic=candidate.task["generation_logic"],
        question_type=candidate.task.get("question_type_id", candidate.task_id),
        answer_field=candidate.task["answer_field"],
        expected_answer=candidate.expected_answer,
        expected_evidence=json.dumps(candidate.expected_evidence, ensure_ascii=False),
        selected_question_template=candidate.question_template,
        candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
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
                "Repair one mapped letter-choice VQA question response. Return JSON only.\n"
                f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
        return _validate(candidate, repaired)


def review(
    candidate: Any, generated: dict[str, Any], _benchmark_info: dict[str, object]
) -> tuple[bool, str]:
    if candidate.expected_answer is None or candidate.expected_evidence is None:
        raise ValueError("Mapped letter review requires a fixed answer and evidence.")
    prompt = VQA_REVIEW_PROMPT.format(
        medical_task_description=candidate.task["medical_task_description"],
        generation_logic=candidate.task["generation_logic"],
        answer_field=candidate.task["answer_field"],
        answer_mapping=json.dumps(candidate.task["answer_value_mapping"], ensure_ascii=False),
        candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
        supporting_data=json.dumps(candidate.supporting_record, ensure_ascii=False),
        selected_question_template=candidate.question_template,
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
