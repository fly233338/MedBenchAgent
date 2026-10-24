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
        if (
            not self.text.strip()
            or hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.digest
        ):
            raise ValueError("Prompt snapshot is empty or does not match its content digest")
        return self


def load_prompt(resource: str) -> str:
    path = PurePosixPath(resource)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".md"
        or "\\" in resource
        or ":" in resource
    ):
        raise ValueError("Expected a package-relative Markdown prompt resource")
    text = files("medbenchagent.agents").joinpath(resource).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty prompt resource: {resource}")
    return text


COMPOSITIONS = {
    "supervisor": ("_shared/prompts/common.md", "supervisor/prompts/system.md"),
    "profile": (
        "_shared/prompts/common.md",
        "data_processing/prompts/base.md",
        "data_processing/prompts/profile.md",
    ),
    "prepare_task_data": (
        "_shared/prompts/common.md",
        "data_processing/prompts/base.md",
        "data_processing/prompts/prepare_task_data.md",
    ),
    "discover_scenarios": (
        "_shared/prompts/common.md",
        "task_planning/prompts/base.md",
        "task_planning/prompts/discover.md",
    ),
    "design_leaves": (
        "_shared/prompts/common.md",
        "task_planning/prompts/base.md",
        "task_planning/prompts/design.md",
    ),
    "generate": ("_shared/prompts/common.md", "generation/prompts/base.md"),
    "review_plan": ("_shared/prompts/common.md", "review/prompts/base.md"),
    "review_candidate": ("_shared/prompts/common.md", "review/prompts/base.md"),
    "coding": ("_shared/prompts/common.md", "data_processing/coding/prompts/base.md"),
    "coding_handoff": ("data_processing/coding/prompts/handoff.md",),
    "structured_repair": ("_shared/prompts/structured_repair.md",),
}


def freeze_composition(name):
    resources = COMPOSITIONS[name] + (
        ("_shared/prompts/structured_repair.md",)
        if name not in {"structured_repair", "coding", "coding_handoff"}
        else ()
    )
    parts = {resource: load_prompt(resource) for resource in resources}
    text = "\n\n".join(parts.values())
    return PromptSnapshot(
        text=text,
        digest=hashlib.sha256(text.encode()).hexdigest(),
        resources={key: hashlib.sha256(value.encode()).hexdigest() for key, value in parts.items()},
    )
