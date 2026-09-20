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


class FeedbackArguments(NoArguments):
    feedback: str = ""


class Delegation(NoArguments):
    objective: NonEmpty
    constraints: list[NonEmpty] = Field(default_factory=list)
    preferences: list[NonEmpty] = Field(default_factory=list)
    revision_notes: str = ""


class DelegationArguments(NoArguments):
    delegation: Delegation


class RunIntent(NoArguments):
    intent: Literal["prepare_dataset", "build_benchmark", "prepare_then_build"]
    prepare_dataset_ids: list[DatasetId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_targets(self):
        if len(self.prepare_dataset_ids) != len(set(self.prepare_dataset_ids)):
            raise ValueError("Preparation targets must be unique")
        if self.intent != "build_benchmark" and not self.prepare_dataset_ids:
            raise ValueError("Preparation intent requires dataset IDs; clarify missing IDs")
        if self.intent == "build_benchmark" and self.prepare_dataset_ids:
            raise ValueError("Use prepare_then_build when new data needs preparation")
        return self




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
    limit: Annotated[int, Field(ge=1, le=10)] = 5


class TableArguments(NoArguments):
    path: NonEmpty
    sheet: str | None = None


class JoinArguments(NoArguments):
    left: NonEmpty
    right: NonEmpty
    left_key: NonEmpty
    right_key: NonEmpty
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
