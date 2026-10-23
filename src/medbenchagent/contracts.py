"""Canonical benchmark item contract at generation and audit boundaries."""
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


CapabilityCategory = Literal["Recognition", "Localization", "Quantification", "Comparison", "Integration", "Generation"]
TaskParadigm = Literal["VQA", "Visual Grounding", "Report Generation", "Detection"]
OutputSpec = Literal["label", "labels", "number", "text", "bbox", "index", "detections"]

class QuestionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["asset", "text", "value"]
    value: Any
    role: str = "input"

    @model_validator(mode="after")
    def shape(self):
        if self.kind == "asset" and (not isinstance(self.value, dict)
                or not isinstance(self.value.get("path"), str) or not self.value["path"]
                or not isinstance(self.value.get("file_digest"), str) or not self.value["file_digest"]):
            raise ValueError("Asset inputs require an explicit file path and digest")
        if self.kind == "text" and not isinstance(self.value, str):
            raise ValueError("Text inputs require a string")
        return self


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    item_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    leaf_task_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    capability_category: CapabilityCategory
    task_paradigm: TaskParadigm
    output_spec: OutputSpec
    question: str = Field(min_length=1)
    template_index: int = Field(default=0, ge=0)
    template_count: int = Field(default=1, ge=1)
    options: list[str] | None = None
    inputs: list[QuestionInput] = Field(min_length=1)
    answer: Any
    input_assets: list[str] = Field(default_factory=list)
    input_evidence_ids: list[str] = Field(default_factory=list)
    reference_evidence_ids: list[str] = Field(default_factory=list)
