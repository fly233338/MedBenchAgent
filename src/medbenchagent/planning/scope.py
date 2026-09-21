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

    @model_validator(mode="after")
    def modality_coverage(self):
        if self.kind == "modality" and not self.modalities:
            raise ValueError("Modality scope must identify the modalities it covers")
        if self.kind == "population" and self.modalities:
            raise ValueError("Population scope cannot declare modalities")
        return self


class ScopeResolution(NoArguments):
    requirement_id: NonEmpty
    status: Literal["implemented", "unsupported"]
    reason: NonEmpty
    conditions: list[SampleCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent(self):
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
    for resolution in resolutions:
        ScopeResolution.model_validate(resolution)
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
