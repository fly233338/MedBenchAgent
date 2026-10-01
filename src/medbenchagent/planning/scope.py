"""Explicit sample scope contracts for benchmark requests."""
from typing import Literal
from pydantic import Field
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty

class ScopeRequirement(NoArguments):
    id: NonEmpty
    description: NonEmpty
    quote: NonEmpty
    kind: Literal["modality", "population"]
    modalities: list[NonEmpty] = Field(default_factory=list)
    operator: Literal["in", "not_in"] = "in"
    logic: Literal["any", "all"] = "any"

class ModalityBinding(NoArguments):
    modality: NonEmpty
    values: list[str | int | float | bool] = Field(min_length=1)
    evidence_references: list[NonEmpty] = Field(default_factory=list)

class ScopeResolution(NoArguments):
    requirement_id: NonEmpty
    status: Literal["implemented", "unsupported"]
    reason: NonEmpty
    conditions: list[dict] = Field(default_factory=list)
    inspection_references: list[NonEmpty] = Field(default_factory=list)
    modality_bindings: list[ModalityBinding] = Field(default_factory=list)
