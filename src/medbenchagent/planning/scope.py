"""Explicit user population requirements and dataset-local executable resolutions."""

from typing import Literal

from pydantic import Field, model_validator

from medbenchagent.planning.protocol import SampleCondition
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty


class ScopeRequirement(NoArguments):
    id: NonEmpty
    description: NonEmpty
    quote: NonEmpty
    kind: Literal["modality", "population"]
    modalities: list[NonEmpty] = Field(default_factory=list)
    operator: Literal["in", "not_in"] = "in"
    logic: Literal["any", "all"] = "any"

    @model_validator(mode="after")
    def modality_coverage(self):
        if self.kind == "modality" and not self.modalities:
            raise ValueError("Modality scope must identify the modalities it covers")
        if self.kind == "population" and self.modalities:
            raise ValueError("Population scope cannot declare modalities")
        return self


class ModalityBinding(NoArguments):
    modality: NonEmpty
    values: list[str | int | float | bool] = Field(min_length=1)
    evidence_references: list[NonEmpty] = Field(default_factory=list)


class ScopeResolution(NoArguments):
    requirement_id: NonEmpty
    status: Literal["implemented", "unsupported"]
    reason: NonEmpty
    conditions: list[SampleCondition] = Field(default_factory=list)
    inspection_references: list[NonEmpty] = Field(default_factory=list)
    modality_bindings: list[ModalityBinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "unsupported" and not self.inspection_references:
            raise ValueError("Unsupported sample scope requires inspection references")
        if bool(self.conditions) != (self.status == "implemented"):
            raise ValueError("Implemented scope requires conditions; unsupported scope must have none")
        if any(c.requirement_id != self.requirement_id for c in self.conditions):
            raise ValueError("Conditions must reference their scope requirement")
        return self


def conditions_for(tree, requirements):
    resolutions = tree.get("scope_resolutions", [])
    expected = {r["id"] for r in requirements}
    if len(resolutions) != len(expected) or {r["requirement_id"] for r in resolutions} != expected:
        raise ValueError("Resolve every sample scope requirement before planning executable leaves")
    by_id = {r["id"]: r for r in requirements}
    for resolution in resolutions:
        parsed = ScopeResolution.model_validate(resolution)
        requested = ScopeRequirement.model_validate(by_id[parsed.requirement_id])
        if parsed.status == "implemented" and requested.kind == "modality":
            if len(parsed.conditions) != 1 or parsed.conditions[0].operator != requested.operator:
                raise ValueError("A modality scope requires exactly one condition with the requested operator")
            if requested.logic == "all" and len(requested.modalities) > 1:
                raise ValueError("Cross-image/multi-sequence conjunction is unsupported by scalar modality scope")
            if ({b.modality for b in parsed.modality_bindings} != set(requested.modalities)
                    or len(parsed.modality_bindings) != len(requested.modalities)):
                raise ValueError("Modality bindings must cover every requested modality exactly once")
            from medbenchagent.answer_mapping import canonical_answer_scalar
            mapped = {canonical_answer_scalar(v) for b in parsed.modality_bindings for v in b.values}
            if mapped != {canonical_answer_scalar(v) for v in parsed.conditions[0].values}:
                raise ValueError("Modality condition differs from the explicit union of bound values")
    if any(r["status"] == "unsupported" for r in resolutions):
        if tree.get("tasks"):
            raise ValueError("Unsupported sample scope cannot publish executable leaves")
        return None
    return [c for r in resolutions for c in r["conditions"]]


def validate_scope_plan(tree, requirements):
    conditions = conditions_for(tree, requirements)
    for leaf in tree["tasks"].values():
        if leaf["protocol"].get("sample_conditions", []) != conditions:
            raise ValueError("Leaf selection differs from the dataset sample scope")
