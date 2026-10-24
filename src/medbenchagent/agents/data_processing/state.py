"""Data Processing state."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from medbenchagent.compilation.artifacts import _within, read_json_artifact, resolve_artifact
from medbenchagent.data.dataset import DatasetSource, file_digest
from medbenchagent.data.task_data import (
    SourceManifest,
    DatasetProfile,
    ProfileReader,
    TaskDataManifest,
    TaskDataReader,
    TaskCase,
    PreparedAsset,
    seal_profile,
    seal_task_data,
    write_once,
)
from medbenchagent.generation.backends import backend_catalog
from medbenchagent.planning.contracts import TaskPlan
from medbenchagent.agents._shared.handoff import (
    WorkState,
    WorkContext,
    binding,
    get_ref,
)


class DataProcessingState(WorkState):
    coding_sessions: list[str]
    coding_calls: dict[str, dict]


@dataclass(frozen=True)
class DataProcessingContext(WorkContext):
    coding: object
    goal: str


def data_contracts():
    return {
        schema.__name__: schema.model_json_schema()
        for schema in (SourceManifest, DatasetProfile, TaskDataManifest, TaskCase, PreparedAsset)
    }


def _allowed_sources(context, call):
    roots = [context.runtime_directory / "coding" / call["request"]["workstream_id"] / "workspace"]
    for ref in context.request.inputs:
        if ref.kind not in {"source_manifest", "dataset_profile", "dataset", "task_data"}:
            continue
        path = resolve_artifact(context.directory, ref)
        value = read_json_artifact(context.directory, ref)
        roots.extend(
            (path.parent / source["root"]).resolve() for source in value.get("sources", [])
        )
    return roots


def _candidate(runtime, call_id, path):
    call = runtime.state.get("coding_calls", {})[call_id]
    receipt = next((row for row in call["files"] if row["path"] == path), None)
    if receipt is None:
        raise ValueError("Select a file returned by this Coding call")
    output = (
        runtime.context.runtime_directory
        / "coding"
        / call["request"]["workstream_id"]
        / "workspace"
        / "outputs"
        / call_id
    )
    actual = _within(output, receipt["path"])
    if actual != Path(receipt["read_path"]).resolve():
        raise ValueError("Coding candidate location changed after execution")

    if file_digest(actual) != receipt["digest"]:
        raise ValueError("Coding candidate changed after execution; make a new Coding call")
    return call, receipt, actual


def validate_candidate(runtime, call_id, path):
    call, receipt, actual = _candidate(runtime, call_id, path)
    schema = {"dataset_profile": DatasetProfile, "task_data": TaskDataManifest}.get(receipt["kind"])
    if schema is None:
        raise ValueError(
            "Submit a dataset_profile or task_data Manifest; supporting files remain referenced members"
        )
    expected = {"profile": "dataset_profile", "prepare_task_data": "task_data"}[
        runtime.context.request.operation
    ]
    if receipt["kind"] != expected:
        raise ValueError(f"This operation publishes {expected}")
    value = schema.model_validate_json(actual.read_bytes())
    allowed = _allowed_sources(runtime.context, call)
    sources = []
    for source in value.sources:
        root = (actual.parent / source.root).resolve()
        if not any(root.is_relative_to(parent.resolve()) for parent in allowed):
            raise ValueError(
                "Candidate source root is outside delegated sources and this Coding workspace"
            )
        sources.append(DatasetSource(source_id=source.source_id, root=str(root)))
    value = value.model_copy(update={"sources": sources})
    if isinstance(value, DatasetProfile):
        bases = [
            read_json_artifact(runtime.context.directory, ref)
            for ref in runtime.context.request.inputs
            if ref.kind == "source_manifest"
        ]
        if not any(base["dataset"] == value.dataset.model_dump() for base in bases):
            raise ValueError("Profile belongs to a different delegated Dataset")
        sealed = seal_profile(value, actual.parent)
    else:
        if value.backend_id not in {row["backend_id"] for row in backend_catalog()}:
            raise ValueError("Unknown TaskData backend")
        plans = [
            TaskPlan.model_validate(read_json_artifact(runtime.context.directory, ref))
            for ref in runtime.context.request.inputs
            if ref.kind == "task_plan"
        ]
        if plans and not any(
            plan.dataset == value.dataset
            and set(value.scenario_ids) <= plan.scenarios.keys()
            and all(
                value.backend_id in plan.scenarios[key].backend_ids for key in value.scenario_ids
            )
            and set(value.requirements) <= plan.requirements.keys()
            for plan in plans
        ):
            raise ValueError(
                "Candidate backend, Scenarios or requirements do not match a delegated TaskPlan"
            )
        profiles = [ref for ref in runtime.context.request.inputs if ref.kind == "dataset_profile"]
        if profiles and value.profile is None:
            raise ValueError("Bind the Profile used to prepare this TaskData")
        if value.profile is not None:
            profile_ref = get_ref(runtime, value.profile.artifact_id, "dataset_profile")
            if binding(profile_ref) != value.profile:
                raise ValueError("TaskData binds a stale Profile")
            ProfileReader(resolve_artifact(runtime.context.directory, profile_ref))
        sealed = seal_task_data(value, actual.parent)
    content = (sealed.model_dump_json(indent=2) + "\n").encode("utf-8")
    verified = actual.parent / ("validated-" + hashlib.sha256(content).hexdigest() + ".json")
    write_once(verified, content)
    reader = (ProfileReader if isinstance(sealed, DatasetProfile) else TaskDataReader)(verified)
    if isinstance(reader, TaskDataReader):
        from medbenchagent.data.task_data import validate_preparation
        from medbenchagent.planning.contracts import DataRequirement

        if not sealed.requirements:
            raise ValueError("TaskData must bind its preparation requirements")
        for key, product in sealed.requirements.items():
            ref = get_ref(runtime, product.artifact_id, "data_requirement")
            requirement = DataRequirement.model_validate(
                read_json_artifact(runtime.context.directory, ref)
            )
            if binding(ref) != product or key != requirement.requirement_id:
                raise ValueError("Stale preparation requirement binding")

            if not any(p.requirements.get(key) == requirement for p in plans):
                raise ValueError("Preparation requirement differs from delegated plan")
            validate_preparation(reader, requirement)
    return sealed, verified, receipt["kind"]
