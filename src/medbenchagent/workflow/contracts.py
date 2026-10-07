"""Supervisor domain tool contracts."""
from typing import Literal
from pydantic import Field
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty, DatasetId

class AllocationPreference(NoArguments):
    id: NonEmpty
    description: NonEmpty
    quote: NonEmpty


class Delegation(NoArguments):
    revision_notes: str = ""


class DelegationArguments(NoArguments):
    delegation: Delegation = Field(default_factory=Delegation)


class ProcessDatasetArguments(NoArguments):
    """Only public data-processing tool contract."""

    dataset_id: DatasetId
    mode: Literal["reuse_or_process", "rebuild"] = "reuse_or_process"
    issue_ids: list[NonEmpty] = Field(default_factory=list)
    notes: str = ""




DATASET_KINDS = frozenset(
    {
        "Dataset",
        "DatasetProfile",
        "DatasetCard",
        "DatasetCandidates",
        "DatasetSelection",
        "Plan",
        "PlanningDelegation",
        "Generation",
        "Allocation",
        "Delivery",
        "PreparedCandidates",
    }
)
