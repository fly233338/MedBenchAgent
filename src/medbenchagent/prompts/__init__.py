"""Static package resources; dynamic context and business policies have other owners."""

import hashlib
from importlib.resources import files
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromptSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    text: str = Field(min_length=1)
    resources: dict[str, str] = Field(default_factory=dict)
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


COMPOSITIONS = {
    "supervisor": ("common.md", "supervisor.md"),
    "profile": ("common.md", "data_processing.md", "data_profile.md"),
    "prepare_task_data": ("common.md", "data_processing.md", "data_prepare.md"),
    "discover_scenarios": ("common.md", "planner.md", "planner_discover.md"),
    "design_leaves": ("common.md", "planner.md", "planner_design.md"),
    "generate": ("common.md", "generation.md"),
    "review_plan": ("common.md", "review.md"),
    "review_candidate": ("common.md", "review.md"),
    "coding": ("common.md", "coding.md"),
    "coding_handoff": ("data_coding_handoff.md",),
    "structured_repair": ("structured_repair.md",),
}


def freeze_composition(name):
    resources = COMPOSITIONS[name] + (("structured_repair.md",) if name not in {"structured_repair", "coding", "coding_handoff"} else ())
    parts = {resource: load_prompt(resource) for resource in resources}
    text = "\n\n".join(parts.values())
    return PromptSnapshot(text=text, digest=hashlib.sha256(text.encode()).hexdigest(),
                          resources={key: hashlib.sha256(value.encode()).hexdigest() for key, value in parts.items()})
