"""Leaf execution contracts and shared annotation eligibility."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from medbenchagent.answer_mapping import canonical_answer_scalar
from medbenchagent.generation.context import (
    scalar_evidence_candidates,
    is_supporting_field_path,
    normalized_bbox,
    project_supporting_record,
    has_meaningful_value,
)
from medbenchagent.metrics import is_metric_supported
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty


class DataBinding(NoArguments):
    supporting_fields: list[NonEmpty] = Field(min_length=1)
    answer_field: NonEmpty
    target_field: NonEmpty | None = None


class AnswerMapping(NoArguments):
    raw_value: str | int | float | bool
    option: NonEmpty


class AnswerContract(NoArguments):
    options: list[NonEmpty] = Field(default_factory=list)
    value_mapping: list[AnswerMapping] = Field(default_factory=list)


class Protocol(NoArguments):
    execution_type: Literal["letter_classification", "bbox_grounding", "free_text"]
    response_format: Literal["letter", "bbox", "free_text"]
    metric: NonEmpty
    data_binding: DataBinding
    answer_contract: AnswerContract
    question_intent: NonEmpty
    medical_constraints: list[NonEmpty] = Field(default_factory=list)
    template_slots: list[Literal["modality", "target"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_contract(self):
        formats = {
            "letter_classification": "letter",
            "bbox_grounding": "bbox",
            "free_text": "free_text",
        }
        if formats[self.execution_type] != self.response_format or not is_metric_supported(
            self.response_format, self.metric
        ):
            raise ValueError("Execution type, response format and metric must agree")
        binding, answer = self.data_binding, self.answer_contract
        fields = binding.supporting_fields
        if len(set(fields)) != len(fields) or not all(is_supporting_field_path(p) for p in fields):
            raise ValueError("Bindings require unique supported annotation paths")
        if binding.answer_field not in fields or (
            binding.target_field and binding.target_field not in fields
        ):
            raise ValueError("Answer and target fields must be included in supporting fields")
        if len(set(self.template_slots)) != len(self.template_slots):
            raise ValueError("Duplicate template slots")
        if binding.target_field:
            if self.response_format != "bbox" or "target" not in self.template_slots:
                raise ValueError("Named bbox targets require the target slot")
            box_parent = binding.answer_field.removesuffix("[*]").rsplit(".", 1)[0]
            if binding.target_field.rsplit(".", 1)[0] != box_parent:
                raise ValueError("Target and bbox must belong to the same annotation entity")
        elif "target" in self.template_slots:
            raise ValueError("Target slot requires an explicit binding")
        if self.response_format == "bbox" and not binding.answer_field.endswith("[*]"):
            raise ValueError("BBox binding must end in a coordinate wildcard")
        if self.response_format == "letter":
            options = answer.options
            if not 2 <= len(options) <= 26 or len(set(options)) != len(options):
                raise ValueError("Classification requires 2-26 unique options")
            values = [canonical_answer_scalar(m.raw_value) for m in answer.value_mapping]
            if None in values or len(set(values)) != len(values):
                raise ValueError("Mapped values must be unique nonempty finite scalars")
            if {m.option for m in answer.value_mapping} != set(options):
                raise ValueError("Every option needs an annotation value mapping")
        elif answer.options or answer.value_mapping:
            raise ValueError("Only classification supports option mappings")
        return self


class TemplateSpecification(NoArguments):
    """The complete information boundary for template generation and review."""

    question_intent: NonEmpty
    medical_constraints: list[NonEmpty]
    execution_type: Literal["letter_classification", "bbox_grounding", "free_text"]
    response_format: Literal["letter", "bbox", "free_text"]
    template_slots: list[Literal["modality", "target"]]
    language: NonEmpty


def template_specification(protocol: dict, language: str) -> TemplateSpecification:
    p = Protocol.model_validate(protocol)
    return TemplateSpecification(
        question_intent=p.question_intent,
        medical_constraints=p.medical_constraints,
        execution_type=p.execution_type,
        response_format=p.response_format,
        template_slots=p.template_slots,
        language=language,
    )


def _resolve_answer(record: dict, protocol: dict) -> dict | None:
    """Resolve one unique annotation witness; shared by counting, generation and audit."""
    binding = protocol["data_binding"]
    if not all(
        has_meaningful_value(project_supporting_record(record, [p]))
        for p in binding["supporting_fields"]
    ):
        return None
    values = scalar_evidence_candidates(record, binding["answer_field"])
    fmt = protocol["response_format"]
    if fmt == "letter":
        mapping = {
            canonical_answer_scalar(m["raw_value"]): m["option"]
            for m in protocol["answer_contract"]["value_mapping"]
        }
        matches = [
            (v, mapping[canonical_answer_scalar(v["value"])])
            for v in values
            if canonical_answer_scalar(v["value"]) in mapping
        ]
        if len(matches) != 1:
            return None
        evidence, option = matches[0]
        return {"answer": option, "evidence": evidence, "target": None}
    if fmt == "free_text":
        if (
            len(values) != 1
            or not isinstance(values[0]["value"], str)
            or not values[0]["value"].strip()
        ):
            return None
        return {"answer": values[0]["value"], "evidence": values[0], "target": None}
    groups = {}
    for value in values:
        indices = value["indices"]
        groups.setdefault(tuple(indices[:-1]), {})[indices[-1]] = value
    boxes = []
    for indices, coordinates in groups.items():
        if set(coordinates) != {0, 1, 2, 3}:
            continue
        evidence = [coordinates[i] for i in range(4)]
        box = normalized_bbox([v["value"] for v in evidence])
        if box is not None and box[0] >= 0 and box[1] >= 0:
            boxes.append((indices, box, evidence))
    if len(boxes) != 1:
        return None
    indices, box, evidence = boxes[0]
    target = None
    if binding.get("target_field"):
        names = [
            v["value"]
            for v in scalar_evidence_candidates(record, binding["target_field"])
            if tuple(v["indices"][: len(indices)]) == indices
        ]
        if len(names) != 1 or not isinstance(names[0], str) or not names[0].strip():
            return None
        target = names[0]
    return {"answer": box, "evidence": evidence, "target": target}


def resolve_eligibility(record: dict, protocol: dict) -> dict:
    """One eligibility boundary for planning, generation, audit and explanations."""
    answer = _resolve_answer(record, protocol)
    if answer is not None:
        return {"eligible": True, "reason": "eligible", "resolved": answer}
    binding = protocol["data_binding"]
    if not all(has_meaningful_value(project_supporting_record(record, [p]))
               for p in binding["supporting_fields"]):
        reason = "missing_field"
    else:
        values = scalar_evidence_candidates(record, binding["answer_field"])
        if protocol["response_format"] == "letter":
            mapped = {canonical_answer_scalar(m["raw_value"])
                      for m in protocol["answer_contract"]["value_mapping"]}
            count = sum(canonical_answer_scalar(v["value"]) in mapped for v in values)
            reason = "unmapped_value" if count == 0 else "multiple_candidates"
        elif protocol["response_format"] == "bbox":
            reason = "no_unique_valid_pair"
        else:
            reason = "multiple_candidates" if len(values) > 1 else "invalid_reference"
    return {"eligible": False, "reason": reason, "resolved": None}


def resolve_answer(record: dict, protocol: dict) -> dict | None:
    return resolve_eligibility(record, protocol)["resolved"]


def validate_dataset_protocol(protocol: dict, records) -> int:
    p = Protocol.model_validate(protocol).model_dump()
    remaining = {canonical_answer_scalar(m["raw_value"])
                 for m in p["answer_contract"]["value_mapping"]}
    count = 0
    for row in records:
        for value in scalar_evidence_candidates(row, p["data_binding"]["answer_field"]):
            remaining.discard(canonical_answer_scalar(value["value"]))
        count += resolve_answer(row, p) is not None
    if remaining:
        raise ValueError("Answer mapping introduces values absent from annotation")
    return count
