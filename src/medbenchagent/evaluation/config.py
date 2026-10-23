"""Post-planning metric configuration for benchmark tasks."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.contracts import CapabilityCategory, OutputSpec, TaskParadigm


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    output_spec: OutputSpec
    task_paradigm: TaskParadigm | None = None
    capability_category: CapabilityCategory | None = None
    parameters: dict[str, object] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "evaluation-config"
    metrics: list[MetricSpec] = Field(default_factory=list)

    def for_task(self, *, output_spec: OutputSpec, task_paradigm: TaskParadigm, capability_category: CapabilityCategory) -> list[MetricSpec]:
        return [metric for metric in self.metrics
                if metric.output_spec == output_spec
                and (metric.task_paradigm is None or metric.task_paradigm == task_paradigm)
                and (metric.capability_category is None or metric.capability_category == capability_category)]

