"""Execution bindings and immutable generation products; Leaf owns assembly semantics."""

from typing import Any
from pydantic import Field, model_validator

from medbenchagent.contracts import BenchmarkItem
from medbenchagent.data.models import CanonicalModel
from medbenchagent.data.task_data import ProductBinding, TaskDataReader
from medbenchagent.planning.contracts import TaskPlan


class GenerationSpec(CanonicalModel):
    task_plan: ProductBinding
    task_id: str = Field(min_length=1)
    task_revision: int = Field(ge=1)
    task_data: ProductBinding
    case_ids: list[str] = Field(default_factory=list)
    sample_count: int = Field(ge=1)
    seed: int = Field(ge=0)


class CaseFailure(CanonicalModel):
    case_id: str
    code: str
    message: str


class GenerationBatch(CanonicalModel):
    spec: ProductBinding
    case_ids: list[str]
    items: list[BenchmarkItem]
    failures: list[CaseFailure] = Field(default_factory=list)


class Allocation(CanonicalModel):
    target: ProductBinding
    requested: int = Field(ge=1)
    counts: dict[str, int]
    capacities: dict[str, int]

    @model_validator(mode="after")
    def quotas(self):
        if (
            set(self.counts) != set(self.capacities)
            or sum(self.counts.values()) > self.requested
            or any(value < 0 or value > self.capacities[key] for key, value in self.counts.items())
        ):
            raise ValueError("Allocation must respect actual capacities and the unchanged target")
        return self


class GenerationCandidate(CanonicalModel):
    allocation: Allocation
    specs: list[ProductBinding]
    batches: list[ProductBinding]
    items_path: str = "items.jsonl"
    quarantine_path: str = "quarantine.jsonl"
    members: dict[str, str]
    counts: dict[str, int]
    task_counts: dict[str, dict[str, int]]
    failures: list[CaseFailure] = Field(default_factory=list)


class AuditReport(CanonicalModel):
    target: ProductBinding
    passed: bool
    checked_count: int
    findings: list[dict[str, Any]]
    counts: dict[str, int]


class ReviewFinding(CanonicalModel):
    object_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    blocking: bool
    summary: str = Field(min_length=1)
    evidence_artifact_ids: list[str] = Field(min_length=1)
    suggested_owner: str = Field(min_length=1)


class ReviewReport(CanonicalModel):
    target: ProductBinding
    audit: ProductBinding | None = None
    findings: list[ReviewFinding]
    checked_task_ids: list[str]
    checked_scenario_ids: list[str] = Field(default_factory=list)
    checked_item_ids: list[str] = Field(default_factory=list)
    unchecked_scope: list[str] = Field(default_factory=list)
    approved: bool
    semantic_sampled: bool = False

    @model_validator(mode="after")
    def coverage(self):
        if any(
            len(values) != len(set(values))
            for values in (self.checked_task_ids, self.checked_scenario_ids, self.checked_item_ids)
        ):
            raise ValueError("Review coverage IDs must be unique")
        if self.approved and (self.unchecked_scope or any(f.blocking for f in self.findings)):
            raise ValueError("Approved Review cannot contain blocking findings or unchecked scope")
        return self


def validate_generation_spec(
    spec, *, plan: TaskPlan, plan_artifact_id: str, plan_digest: str, reader: TaskDataReader
):
    from .backends import validate_leaf

    spec = GenerationSpec.model_validate(spec)
    if spec.task_plan != ProductBinding(artifact_id=plan_artifact_id, digest=plan_digest):
        raise ValueError("GenerationSpec binds another Task Plan")
    task = plan.tasks[spec.task_id]
    if (
        task.task_id in plan.exclusions
        or task.feasibility != "feasible"
        or task.revision != spec.task_revision
    ):
        raise ValueError("Task revision is excluded or not ready")
    if task.task_data != spec.task_data or reader.content_digest != spec.task_data.digest:
        raise ValueError("GenerationSpec binds another TaskData product")
    acceptance = plan.task_data_acceptances.get(spec.task_data.artifact_id)
    if acceptance is None or acceptance.digest != spec.task_data.digest:
        raise ValueError("Planner has not accepted this TaskData")
    if set(task.requirement_ids) - plan.acceptances.keys():
        raise ValueError("Leaf has unaccepted data requirements")
    if task.dataset != reader.manifest.dataset:
        raise ValueError("TaskData belongs to another source dataset revision")
    if plan.scenarios[task.scenario_id].decision != "keep":
        raise ValueError("Task Scenario is not kept")
    if (
        task.assembly is not None
        and task.assembly.backend_id not in plan.scenarios[task.scenario_id].backend_ids
    ):
        raise ValueError("Leaf backend is not selected by its Scenario")
    validate_leaf(task, reader)
    if len(spec.case_ids) != len(set(spec.case_ids)):
        raise ValueError("Duplicate selected Cases")
    eligible = {case.case_id for case in reader.cases if task.scenario_id in case.scenario_ids}
    if set(spec.case_ids) - eligible:
        raise ValueError("Selected Cases do not belong to this Scenario")
    return spec
