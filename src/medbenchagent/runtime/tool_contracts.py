"""Validated JSON arguments shared by the agent tool boundaries."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


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


class PrepareDataArguments(FeedbackArguments):
    source_root: NonEmpty
    dataset_id: DatasetId


class SelectDatasetArguments(NoArguments):
    dataset_id: DatasetId | None = None
    annotation_path: NonEmpty | None = None


class ProfileArguments(NoArguments):
    revision: NonEmpty = "1"
    refresh: bool = False


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


class MappingArguments(NoArguments):
    sources: Annotated[list[NonEmpty], Field(min_length=1)]
    authoritative_population: NonEmpty
    expected_images: Annotated[int, Field(gt=0)]
    row_granularity: dict[str, NonEmpty]
    entity_fields: dict[Literal["patient", "case", "study", "image"], dict[str, str]]
    joins: list[NonEmpty]
    identifiers_and_order: NonEmpty
    bbox_conventions: NonEmpty
    modality_anatomy: NonEmpty
    conflicts_and_unmatched: NonEmpty
    excluded_fields: dict[str, str]
    unresolved: list[str]
    decision_references: list[str] = []


class ScriptArguments(NoArguments):
    code: Annotated[str, Field(min_length=1)]


class ImagePolicy(NoArguments):
    encoding: Literal["preserve", "window"] = "preserve"
    low: float | None = None
    high: float | None = None
    invert: bool = False
    decision_reference: str = ""


class ImagePlanArguments(NoArguments):
    overrides: dict[str, ImagePolicy] = Field(default_factory=dict)
