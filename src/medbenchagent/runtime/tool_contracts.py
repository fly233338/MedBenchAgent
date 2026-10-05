"""Validated JSON arguments shared by active agent tool boundaries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DatasetId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AskUserArguments(NoArguments):
    question: NonEmpty
    reference: str = ""
    purpose: str = ""
    candidate_snapshot_path: str = ""
    allowed_dataset_ids: list[NonEmpty] = Field(default_factory=list)
    allowed_extension_ids: list[NonEmpty] = Field(default_factory=list)
    scope: Literal["local", "global"] = "local"
    issue_id: str = ""
    subject_scope: Literal["dataset", "source", "multi_source"] | None = Field(
        default=None,
        description="Dataset scope has no source IDs; source needs one selected source; multi_source needs at least two.",
    )
    source_ids: list[NonEmpty] = Field(
        default_factory=list,
        description="Source IDs for the subject scope. A selected active source can be inferred for source scope.",
    )

    @model_validator(mode="after")
    def validate_source_set(self):
        count = len(set(self.source_ids))
        if self.subject_scope == "dataset" and count:
            raise ValueError("Dataset clarification must not include source_ids")
        if self.subject_scope == "source" and count > 1:
            raise ValueError("Source clarification requires exactly one source_id")
        if self.subject_scope == "multi_source" and count < 2:
            raise ValueError("Multi-source clarification requires at least two source_ids")
        return self


class FeedbackArguments(NoArguments):
    feedback: str = ""


class AllocationPreference(NoArguments):
    id: NonEmpty
    description: NonEmpty
    quote: NonEmpty


class Delegation(NoArguments):
    revision_notes: str = ""


class DelegationArguments(NoArguments):
    delegation: Delegation = Field(default_factory=Delegation)


class PrepareDataArguments(FeedbackArguments):
    dataset_id: DatasetId
    mode: Literal["reuse_or_prepare", "rebuild"] = "reuse_or_prepare"


class PageArguments(NoArguments):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)
