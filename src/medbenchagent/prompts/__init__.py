"""Static package resources; dynamic context and business policies have other owners."""

import hashlib
from importlib.resources import files
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    text: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def check_content(self):
        if not self.text.strip() or hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.digest:
            raise ValueError("Prompt snapshot is empty or does not match its content digest")
        return self


def load_prompt(resource: str) -> str:
    path = PurePosixPath(resource)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".md" or "\\" in resource or ":" in resource:
        raise ValueError("Expected a package-relative Markdown prompt resource")
    text = files(__package__).joinpath(resource).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty prompt resource: {resource}")
    return text


def freeze_prompt(resource: str) -> PromptSnapshot:
    text = load_prompt(resource)
    return PromptSnapshot(text=text, digest=hashlib.sha256(text.encode("utf-8")).hexdigest())
