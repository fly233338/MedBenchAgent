"""Validated JSON arguments shared by the agent tool boundaries."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DatasetId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
NonNegative = Annotated[int, Field(ge=0)]


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AskUserArguments(NoArguments):
    question: NonEmpty


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


class ReviseTaskArguments(NoArguments):
    task_id: NonEmpty
    feedback: NonEmpty


class AllocateArguments(NoArguments):
    counts: dict[str, NonNegative] | None = None


class MaterialsArguments(NoArguments):
    files: Annotated[list[NonEmpty], Field(min_length=1)]


class MedicalDesignArguments(FeedbackArguments):
    task_id: NonEmpty


class InspectFilesArguments(NoArguments):
    directory: str = ""
    offset: NonNegative = 0


class ReadTextArguments(NoArguments):
    path: NonEmpty
    offset: NonNegative = 0


class WriteScriptArguments(NoArguments):
    code: Annotated[str, Field(min_length=1)]
    mapping: NonEmpty
