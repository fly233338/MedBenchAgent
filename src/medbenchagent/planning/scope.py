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
