"""Typed deterministic Reference Resolver for annotation.v5 cases."""

from __future__ import annotations

import math
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.planning.task_space import OutputSpec, EvidenceSelector


class ResolvedTaskItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_evidence_ids: list[str]
    reference: Any
    reference_evidence_ids: list[str]


class NotApplicable(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=1)


class ReadStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["read"] = "read"
    selector: EvidenceSelector


class CollectStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["collect"] = "collect"
    selector: EvidenceSelector


class SelectStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["select"] = "select"
    input: str = Field(min_length=1)
    index: int | None = None
    value: Any = None


class CompareStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["compare"] = "compare"
    left: str = Field(min_length=1)
    right: str = Field(min_length=1)
    mode: Literal["equal", "less", "greater", "difference"] = "equal"


class ComputeStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["compute"] = "compute"
    inputs: list[str] = Field(min_length=1)
    operation: Literal["sum", "mean", "difference", "ratio", "count"]


class LocateStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["locate"] = "locate"
    input: str = Field(min_length=1)


class ReferenceTextStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["reference_text"] = "reference_text"
    selector: EvidenceSelector


ResolverStep = Union[ReadStep, CollectStep, SelectStep, CompareStep, ComputeStep, LocateStep, ReferenceTextStep]


class ResolverSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: list[EvidenceSelector] = Field(min_length=1)
    steps: list[ResolverStep] = Field(min_length=1)
    output_spec: OutputSpec
    result_step: str = Field(min_length=1)

    @classmethod
    def parse(cls, value: Any) -> "ResolverSpec":
        return cls.model_validate(value)


def _evidence(case: dict[str, Any], selector: EvidenceSelector) -> list[dict[str, Any]]:
    rows = []
    for item in case.get("evidence", []):
        if selector.name is not None and item.get("name") != selector.name:
            continue
        if selector.evidence_type is not None and item.get("type") != selector.evidence_type:
            continue
        if selector.relation is not None and selector.relation not in item.get("ref", []):
            continue
        rows.append(item)
    return rows


def _scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Resolver requires finite numeric evidence")
    return float(value)


def resolve(case: dict[str, Any], resolver_spec: ResolverSpec | dict[str, Any]) -> ResolvedTaskItem | NotApplicable:
    spec = ResolverSpec.parse(resolver_spec)
    values: dict[str, Any] = {}
    value_refs: dict[str, set[str]] = {}
    input_ids = [item["id"] for selector in spec.inputs for item in _evidence(case, selector)]
    if not input_ids:
        return NotApplicable(reason="missing_input_evidence")
    for index, step in enumerate(spec.steps):
        key = f"step_{index}"
        if isinstance(step, (ReadStep, CollectStep, ReferenceTextStep)):
            rows = _evidence(case, step.selector)
            if not rows:
                return NotApplicable(reason=f"missing_{step.op}_evidence")
            values[key] = rows if isinstance(step, CollectStep) else rows[0]
            value_refs[key] = {row["id"] for row in rows}
        elif isinstance(step, SelectStep):
            source = values.get(step.input)
            if not isinstance(source, list) or not source:
                return NotApplicable(reason="select_source_missing")
            selected = source[step.index] if step.index is not None and -len(source) <= step.index < len(source) else None
            if step.value is not None:
                selected = next((x for x in source if x.get("value") == step.value), None)
            if selected is None:
                return NotApplicable(reason="select_no_match")
            values[key] = selected
            value_refs[key] = {selected["id"]} if isinstance(selected, dict) and "id" in selected else set(value_refs.get(step.input, set()))
        elif isinstance(step, CompareStep):
            left, right = values.get(step.left), values.get(step.right)
            if left is None or right is None:
                return NotApplicable(reason="compare_input_missing")
            lv, rv = left.get("value", left), right.get("value", right)
            try:
                values[key] = {"equal": lv == rv, "less": lv < rv, "greater": lv > rv, "difference": _scalar(lv) - _scalar(rv)}[step.mode]
            except (TypeError, ValueError):
                return NotApplicable(reason="compare_invalid_values")
            value_refs[key] = value_refs.get(step.left, set()) | value_refs.get(step.right, set())
        elif isinstance(step, ComputeStep):
            try:
                numbers = [_scalar(values[name].get("value", values[name])) for name in step.inputs]
            except (KeyError, TypeError, ValueError):
                return NotApplicable(reason="compute_invalid_values")
            if step.operation == "sum":
                result = sum(numbers)
            elif step.operation == "mean":
                result = sum(numbers) / len(numbers)
            elif step.operation == "difference":
                result = numbers[0] - numbers[1]
            elif step.operation == "ratio":
                if numbers[1] == 0:
                    return NotApplicable(reason="compute_division_by_zero")
                result = numbers[0] / numbers[1]
            else:
                result = len(numbers)
            values[key] = result
            value_refs[key] = set().union(*(value_refs.get(name, set()) for name in step.inputs))
        elif isinstance(step, LocateStep):
            source = values.get(step.input)
            values[key] = source.get("value", source) if source is not None else None
            value_refs[key] = set(value_refs.get(step.input, set()))
        if values.get(key) is None:
            return NotApplicable(reason=f"{step.op}_not_applicable")
    reference = values.get(spec.result_step)
    if reference is None:
        return NotApplicable(reason="result_missing")
    ref_ids = sorted(value_refs.get(spec.result_step, set()))
    if isinstance(reference, dict) and "id" in reference:
        ref_ids = sorted(set(ref_ids) | {reference["id"]})
    if set(input_ids) & set(ref_ids):
        return NotApplicable(reason="input_reference_leakage")
    return ResolvedTaskItem(input_evidence_ids=input_ids, reference=reference, reference_evidence_ids=ref_ids)

