"""Canonical benchmark item contract at generation and audit boundaries."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    item_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    leaf_task_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    capability_category: Literal["Recognition", "Localization", "Quantification", "Comparison", "Integration", "Generation"]
    task_paradigm: Literal["VQA", "Visual Grounding", "Report Generation"]
    output_spec: Literal["label", "labels", "number", "text", "bbox", "index"]
    question: str = Field(min_length=1)
    template_index: int = Field(default=0, ge=0)
    template_count: int = Field(default=1, ge=1)
    options: list[str] | None = None
    input_images: list[dict[str, Any]] = Field(min_length=1)
    sampling_unit_id: str | None = Field(default=None, min_length=1)
    split: str | None = None
    view_ids: list[str] = Field(default_factory=list)
    panel_id: str | None = None
    sampling_order: int | None = Field(default=None, ge=0)
    answer: Any
    input_bindings: list[dict[str, Any]] = Field(default_factory=list)
    input_assets: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)
