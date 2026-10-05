"""Canonical source identity contracts used by preparation and planning."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(min_length=1)]
Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Source(Contract):
    id: Name
    version: Text | None = None
    kind: Literal["raw", "annotation"] = "raw"
    root: Text
    annotation: str = "cases.jsonl"


class SourcesArguments(Contract):
    sources: list[Source] = Field(min_length=1)
    dataset_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    feedback: str = ""

    @model_validator(mode="after")
    def unique_sources(self):
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("Source IDs must be unique within a dataset")
        return self
