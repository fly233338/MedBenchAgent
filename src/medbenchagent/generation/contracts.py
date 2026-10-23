"""Executable closed-VQA configuration. No generator, Resolver DSL or runtime state."""

from string import Formatter
from typing import Literal

from pydantic import Field, model_validator

from medbenchagent.data.dataset import DatasetRef
from medbenchagent.data.models import CanonicalModel
from medbenchagent.planning.semantic import validate_plan


class ProductBinding(CanonicalModel):
    artifact_id: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceSelection(CanonicalModel):
    name: str = Field(min_length=1)
    evidence_type: Literal["asset", "text", "value"]


class GenerationSpec(CanonicalModel):
    task_plan: ProductBinding
    task_id: str = Field(min_length=1)
    task_revision: int = Field(ge=1)
    dataset: DatasetRef
    dataset_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_family: Literal["closed_vqa"] = "closed_vqa"
    case_ids: list[str] = Field(min_length=1)
    inputs: list[EvidenceSelection] = Field(min_length=1)
    reference_field: str = Field(min_length=1)
    derived_data: ProductBinding | None = None
    reference_rule: Literal["identity", "label_map"] = "identity"
    label_map: dict[str, str] = Field(default_factory=dict)
    output_format: Literal["label"] = "label"
    backend: Literal["closed_vqa_template"] = "closed_vqa_template"
    templates: list[str] = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    sample_count: int = Field(ge=1)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def closed_output(self):
        if len(self.case_ids) != len(set(self.case_ids)) or len(self.options) != len(
            set(self.options)
        ):
            raise ValueError("Case IDs and options must be unique")
        if len(self.inputs) != len({(item.name, item.evidence_type) for item in self.inputs}):
            raise ValueError("Input selectors must be unique")
        if any(not item.strip() for item in [*self.options, *self.templates]):
            raise ValueError("Options and templates cannot be empty")
        if self.reference_rule == "label_map":
            if not self.label_map or set(self.label_map.values()) - set(self.options):
                raise ValueError("label_map must map into the declared options")
        elif self.label_map:
            raise ValueError("identity reference rule cannot contain a label_map")
        for template in self.templates:
            for _, name, format_spec, conversion in Formatter().parse(template):
                if name is not None and (
                    name not in {"options", "task_objective"} or format_spec or conversion
                ):
                    raise ValueError("Templates may only interpolate options and task_objective")
        return self


def validate_generation_spec(
    spec, *, plan, plan_artifact_id, plan_digest, reader, derived_views=None
):
    """Validate bindings and execution inputs, without claiming backend readiness or generating items.

    derived_views contains Host-verified artifact_id -> (digest, decoded canonical derived_view).
    The Host must read these through the formal artifact reader before calling this function.
    """
    spec = GenerationSpec.model_validate(spec)
    validate_plan(plan, reader, publish=True)
    if spec.task_plan.artifact_id != plan_artifact_id or spec.task_plan.digest != plan_digest:
        raise ValueError("GenerationSpec does not bind the supplied Task Plan artifact")
    task = plan.tasks[spec.task_id]
    if (
        task.revision != spec.task_revision
        or task.task_family != spec.task_family
        or task.feasibility != "feasible"
        or task.task_id in plan.exclusions
        or task.output_paradigm != "label"
    ):
        raise ValueError(
            "Task revision, family or semantic feasibility is not executable by this spec"
        )
    if (
        spec.dataset != plan.dataset
        or spec.dataset_digest != plan.dataset_digest
        or reader.content_digest != spec.dataset_digest
    ):
        raise ValueError("GenerationSpec binds a different Dataset snapshot")
    expected_inputs = {
        (
            reader.get_evidence(ref.case_id, ref.evidence_id).name,
            reader.get_evidence(ref.case_id, ref.evidence_id).type,
        )
        for ref in task.input_evidence_refs
    }
    if {(item.name, item.evidence_type) for item in spec.inputs} != expected_inputs:
        raise ValueError("Execution inputs do not match the task's input evidence semantics")
    derived = None
    if spec.derived_data is not None:
        entry = (derived_views or {}).get(spec.derived_data.artifact_id)
        if entry is None or entry[0] != spec.derived_data.digest:
            raise ValueError("Missing or stale derived data")
        value = entry[1]
        requirement_id = value.get("requirement_id")
        acceptance = plan.acceptances.get(requirement_id)
        if (
            requirement_id not in task.requirement_ids
            or acceptance is None
            or acceptance.artifact_id != spec.derived_data.artifact_id
            or acceptance.digest != spec.derived_data.digest
            or value.get("dataset") != spec.dataset.model_dump()
        ):
            raise ValueError("Derived data was not accepted for this task and Dataset")
        rows = value.get("rows", [])
        derived = {row["case_id"]: row for row in rows}
        if len(derived) != len(rows):
            raise ValueError("Duplicate derived Cases")
    for case_id in spec.case_ids:
        case = reader.get_case(case_id)
        chosen = set()
        for selector in spec.inputs:
            matches = [
                e
                for e in case.evidence
                if e.name == selector.name and e.type == selector.evidence_type
            ]
            if not matches or any(e.input_visibility != "model_visible" for e in matches):
                raise ValueError("Generation inputs are missing or not model_visible")
            chosen.update(e.id for e in matches)
        if derived is None:
            names = {
                reader.get_evidence(ref.case_id, ref.evidence_id).name
                for ref in task.reference_evidence_refs
            }
            if spec.reference_field not in names:
                raise ValueError("Reference field does not match the task evidence")
            references = [e for e in case.evidence if e.name == spec.reference_field]
            if len(references) != 1 or references[0].id in chosen:
                raise ValueError("Reference must resolve uniquely and cannot be a model input")
            answer = references[0].value
        else:
            answer = derived[case_id][spec.reference_field]
        label = spec.label_map.get(str(answer)) if spec.reference_rule == "label_map" else answer
        if not isinstance(label, str) or label not in spec.options:
            raise ValueError(f"Reference is outside the closed options for Case {case_id}")
    return spec
