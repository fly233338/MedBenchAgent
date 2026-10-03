"""Typed deterministic Reference Resolver for annotation cases."""

from __future__ import annotations

import math
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    result_step: str = Field(
        min_length=1,
        description="ID of the result step, using positional names step_0, step_1, step_2, ...",
    )

    @model_validator(mode="after")
    def compile(self):
        roles = {s.role for s in self.inputs}
        if len(roles) != len(self.inputs):
            raise ValueError("Input roles must be unique")
        for selector in self.inputs:
            if selector.relation is not None or selector.evidence_type not in {"asset", "text", "value", "region"} or not selector.name:
                raise ValueError(
                    "Resolver inputs require explicit asset/text/value/region names "
                    "and cannot depend on reference relations"
                )
        seen = set()
        for index, step in enumerate(self.steps):
            dependencies = ([step.input] if isinstance(step, (SelectStep, LocateStep)) else
                            [step.left, step.right] if isinstance(step, CompareStep) else
                            step.inputs if isinstance(step, ComputeStep) else [])
            if not set(dependencies) <= seen:
                raise ValueError("Resolver steps must reference earlier steps")
            if isinstance(step, SelectStep) and ((step.index is None) == (step.value is None)):
                raise ValueError("Select requires exactly one index or value")
            if isinstance(step, ComputeStep) and step.operation in {"difference", "ratio"} and len(step.inputs) != 2:
                raise ValueError("Binary compute requires two inputs")
            if hasattr(step, "selector") and step.selector.relation and step.selector.relation not in roles:
                raise ValueError("Reference relation must name a model input role")
            seen.add(f"step_{index}")
        if self.result_step not in seen:
            available = ", ".join(sorted(seen))
            raise ValueError(
                f"Unknown result step {self.result_step!r}; "
                f"result_step must be one of: {available}"
            )
        return self

    @classmethod
    def parse(cls, value: Any) -> "ResolverSpec":
        return cls.model_validate(value)


def validate_resolver_policy(spec: ResolverSpec | dict[str, Any], policy: dict[str, Any]) -> ResolverSpec:
    """Apply the experiment Registry policy after structural parsing."""
    parsed = ResolverSpec.parse(spec)
    allowed_steps = set(policy.get("step_operations", []))
    allowed_compute = set(policy.get("compute_operations", []))
    allowed_compare = set(policy.get("compare_modes", []))
    allowed_types = set(policy.get("input_evidence_types", []))
    for selector in parsed.inputs:
        if selector.evidence_type not in allowed_types:
            raise ValueError(f"Resolver input evidence type is disabled by the Registry: {selector.evidence_type}")
    for step in parsed.steps:
        if step.op not in allowed_steps:
            raise ValueError(f"Resolver operation is disabled by the Registry: {step.op}")
        if isinstance(step, ComputeStep) and step.operation not in allowed_compute:
            raise ValueError(f"Resolver compute operation is disabled by the Registry: {step.operation}")
        if isinstance(step, CompareStep) and step.mode not in allowed_compare:
            raise ValueError(f"Resolver compare mode is disabled by the Registry: {step.mode}")
    return parsed


def _evidence(case: dict[str, Any], selector: EvidenceSelector, roles=None) -> list[dict[str, Any]]:
    rows = []
    for item in case.get("evidence", []):
        if selector.name is not None and item.get("name") != selector.name:
            continue
        if selector.evidence_type is not None and item.get("type") != selector.evidence_type:
            continue
        if selector.unit is not None and item.get("unit") != selector.unit:
            continue
        if selector.relation is not None and not set(item.get("ref", [])) & set((roles or {}).get(selector.relation, [])):
            continue
        rows.append(item)
    return rows


def _scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Resolver requires finite numeric evidence")
    return float(value)


def validate_output(value, spec):
    """The same strict output boundary is used by resolution, generation and audit."""
    valid = {
        OutputSpec.LABEL: lambda: type(value) in (str, int, bool) and value != "",
        OutputSpec.LABELS: lambda: isinstance(value, list) and all(type(x) is str and x for x in value) and len(set(value)) == len(value),
        OutputSpec.NUMBER: lambda: type(value) in (int, float) and math.isfinite(value),
        OutputSpec.TEXT: lambda: isinstance(value, str) and bool(value.strip()),
        OutputSpec.BBOX: lambda: isinstance(value, list) and len(value) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in value) and 0 <= value[0] < value[2] <= 1 and 0 <= value[1] < value[3] <= 1,
        OutputSpec.INDEX: lambda: type(value) is int and value >= 0,
    }
    if not valid[OutputSpec(spec)]():
        raise ValueError("Invalid reference for OutputSpec")
    return value


def _value(value):
    if isinstance(value, dict):
        return value["value"]
    if isinstance(value, list):
        return [_value(item) for item in value]
    return value


def resolve(case: dict[str, Any], resolver_spec: ResolverSpec | dict[str, Any]) -> ResolvedTaskItem | NotApplicable:
    spec = ResolverSpec.parse(resolver_spec)
    values: dict[str, Any] = {}
    value_refs: dict[str, set[str]] = {}
    roles = {selector.role: [item["id"] for item in _evidence(case, selector)] for selector in spec.inputs}
    input_ids = sorted({identifier for ids in roles.values() for identifier in ids})
    if any(not ids for ids in roles.values()):
        return NotApplicable(reason="missing_input_evidence")
    for index, step in enumerate(spec.steps):
        key = f"step_{index}"
        if isinstance(step, (ReadStep, CollectStep, ReferenceTextStep)):
            rows = _evidence(case, step.selector, roles)
            if not rows:
                return NotApplicable(reason=f"missing_{step.op}_evidence")
            if not isinstance(step, CollectStep) and len(rows) != 1:
                return NotApplicable(reason="multiple_matches")
            if isinstance(step, ReferenceTextStep) and any(row["type"] != "text" for row in rows):
                return NotApplicable(reason="invalid_reference_text")
            values[key] = rows if isinstance(step, CollectStep) else rows[0]
            value_refs[key] = {row["id"] for row in rows}
        elif isinstance(step, SelectStep):
            source = values.get(step.input)
            if not isinstance(source, list) or not source:
                return NotApplicable(reason="select_source_missing")
            selected = source[step.index] if step.index is not None and -len(source) <= step.index < len(source) else None
            if step.value is not None:
                matches = [x for x in source if _value(x) == step.value]
                selected = matches[0] if len(matches) == 1 else None
            if selected is None:
                return NotApplicable(reason="select_no_match")
            values[key] = selected
            value_refs[key] = {selected["id"]} if isinstance(selected, dict) and "id" in selected else set(value_refs.get(step.input, set()))
        elif isinstance(step, CompareStep):
            left, right = values.get(step.left), values.get(step.right)
            if left is None or right is None:
                return NotApplicable(reason="compare_input_missing")
            lv, rv = _value(left), _value(right)
            try:
                if step.mode == "equal":
                    values[key] = type(lv) is type(rv) and lv == rv
                elif step.mode == "less":
                    values[key] = _scalar(lv) < _scalar(rv)
                elif step.mode == "greater":
                    values[key] = _scalar(lv) > _scalar(rv)
                else:
                    values[key] = _scalar(lv) - _scalar(rv)
            except (TypeError, ValueError):
                return NotApplicable(reason="compare_invalid_values")
            value_refs[key] = value_refs.get(step.left, set()) | value_refs.get(step.right, set())
        elif isinstance(step, ComputeStep):
            try:
                numbers = [_scalar(_value(values[name])) for name in step.inputs] if step.operation != "count" else []
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
                result = sum(len(values[name]) if isinstance(values[name], list) else 1 for name in step.inputs)
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
    by_id = {row["id"]: row for row in case["evidence"]}
    def ancestors(identifier, seen=None):
        seen = set() if seen is None else seen
        if identifier in seen:
            return seen
        seen.add(identifier)
        for parent in by_id[identifier].get("ref", []):
            ancestors(parent, seen)
        return seen
    if any(ancestors(identifier) & set(ref_ids) for identifier in input_ids):
        return NotApplicable(reason="input_reference_leakage")
    units = {by_id[identifier].get("unit") for identifier in ref_ids}
    if any(isinstance(step, (ComputeStep, CompareStep)) for step in spec.steps) and len(units) > 1:
        return NotApplicable(reason="incompatible_units")
    try:
        reference = validate_output(_value(reference), spec.output_spec)
    except ValueError:
        return NotApplicable(reason="invalid_output")
    return ResolvedTaskItem(input_evidence_ids=input_ids, reference=reference, reference_evidence_ids=ref_ids)

