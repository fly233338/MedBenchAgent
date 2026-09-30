"""Validated JSON arguments shared by the agent tool boundaries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DatasetId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
NonNegative = Annotated[int, Field(ge=0)]


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AskUserArguments(NoArguments):
    question: NonEmpty
    reference: str = ""
    scope: Literal["local", "global"] = "local"
    issue_id: str = ""
    subject_scope: Literal["dataset", "source", "multi_source"] | None = Field(
        default=None, description="Dataset scope has no source IDs; source needs one selected source; multi_source needs at least two.")
    source_ids: list[NonEmpty] = Field(
        default_factory=list, description="Source IDs for the subject scope. A selected active source can be inferred for source scope.")
    image_ids: list[NonEmpty] = Field(default_factory=list)

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


class MaterialsArguments(NoArguments):
    files: list[NonEmpty]


class InspectFilesArguments(NoArguments):
    directory: str = ""
    offset: NonNegative = 0


class ReadTextArguments(NoArguments):
    path: NonEmpty
    offset: NonNegative = 0


class PageArguments(NoArguments):
    offset: NonNegative = 0
    limit: Annotated[int, Field(ge=1, le=50)] = 20
    query: str = ""


class FieldArguments(NoArguments):
    fields: Annotated[list[NonEmpty], Field(min_length=1, max_length=30)]


class MaterialArguments(NoArguments):
    file: NonEmpty


class SearchArguments(NoArguments):
    query: NonEmpty
    limit: Annotated[int, Field(ge=1, le=10)] | None = None


class TableArguments(NoArguments):
    path: NonEmpty
    sheet: str | None = None
    delimiter: NonEmpty = ","


class JoinArguments(NoArguments):
    left: NonEmpty
    right: NonEmpty
    left_keys: list[NonEmpty] = Field(min_length=1)
    right_keys: list[NonEmpty] = Field(min_length=1)
    cardinality: Literal["one_to_one", "many_to_one", "one_to_many"]
    how: Literal["left", "inner"] = "left"
    null_keys: Literal["error", "unmatched"] = "error"
    unmatched: Literal["error", "retain", "exclude"] = "error"
    left_delimiter: NonEmpty = ","
    right_delimiter: NonEmpty = ","
    left_sheet: str | None = None
    right_sheet: str | None = None


class ImagePolicy(NoArguments):
    encoding: Literal["preserve", "window"] = "preserve"
    low: float | None = None
    high: float | None = None
    invert: bool = False
    decision_reference: str = ""


class ImagePlanArguments(NoArguments):
    overrides: dict[str, ImagePolicy] = Field(default_factory=dict)


class DisplayPolicy(NoArguments):
    encoding: Literal["window"]
    low: float = Field(allow_inf_nan=False)
    high: float = Field(allow_inf_nan=False)
    invert: bool

    @model_validator(mode="after")
    def interval(self):
        if self.low >= self.high:
            raise ValueError("Display window requires low < high")
        return self


class DisplayDecision(NoArguments):
    policies: dict[str, DisplayPolicy]
