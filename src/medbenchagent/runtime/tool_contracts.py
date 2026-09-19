"""Validated JSON arguments shared by the agent tool boundaries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from medbenchagent.data.contracts import SourcesArguments


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




class PrepareDataArguments(SourcesArguments):
    pass


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
