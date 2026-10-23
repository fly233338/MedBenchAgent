"""Task Planning state."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from langchain.agents import AgentState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import Field
from medbenchagent.compilation.artifacts import (
    read_json_artifact,
    resolve_artifact,
    register_file,
)
from medbenchagent.compilation.contracts import (
    ArtifactRef,
    Contract,
    WorkIssue,
    WorkNeed,
    WorkOutcome,
    WorkRequest,
    WorkResult,
)
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.data.task_data import ProfileReader, TaskDataReader
from medbenchagent.planning.contracts import (
    DataRequirement,
    TaskPlan,
)
from medbenchagent.planning import semantic



class PlannerState(AgentState):
    active_request_id: str
    draft_path: str
    submitted: list[dict]


class PlannerResponse(Contract):
    outcome: WorkOutcome
    summary: str = Field(min_length=1)
    artifact_ids: list[str] = Field(default_factory=list)
    issues: list[WorkIssue] = Field(default_factory=list)
    user_question: str | None = None


@dataclass(frozen=True)
class PlannerContext:
    directory: Path
    request: WorkRequest
    reader: FileDatasetReader | ProfileReader
    dataset: ArtifactRef
    settings: object
    instructions: dict[str, str]
    task_data: dict[str, TaskDataReader] = field(default_factory=dict)
    agent_id: str = "task_planner"

    @property
    def compilation_id(self):
        return self.request.compilation_id

    @property
    def invocation_id(self):
        return self.request.request_id


def _load(state, context):
    path = (context.directory / state["draft_path"]).resolve()
    if not path.is_relative_to(context.directory / "planner" / "drafts"):
        raise ValueError("Draft reference escaped the Planner directory")
    content = path.read_bytes()
    if path.stem != hashlib.sha256(content).hexdigest():
        raise ValueError("Planner draft content changed")
    return TaskPlan.model_validate_json(content)


def _save(plan, context):
    content = (plan.model_dump_json(indent=2) + "\n").encode("utf-8")
    path = (
        context.directory / "planner" / "drafts" / (hashlib.sha256(content).hexdigest() + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Cannot overwrite an immutable Planner draft")
    else:
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return path.relative_to(context.directory).as_posix()


def _product_digest(value):
    content = json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _bounded(value, ceiling):
    encoded = json.dumps(value, ensure_ascii=False)
    if len(encoded) <= ceiling:
        return value
    return {
        "preview": encoded[:ceiling],
        "truncated": True,
        "total_chars": len(encoded),
        "instruction": "Narrow the query or use text_offset to read the next text segment.",
    }


def _update(runtime, base_revision, operation):
    context = runtime.context
    plan = _load(runtime.state, context)
    if plan.revision != base_revision:
        raise ValueError(f"Stale draft revision: expected {plan.revision}")
    before = plan.model_copy(deep=True)
    operation(plan)
    if context.request.operation == "design_leaves" and (plan.axis_assessments != before.axis_assessments or plan.scenarios != before.scenarios):
        raise ValueError("Scenario changes require discover_scenarios")
    if all(scope.kind != "artifact" for scope in context.request.scope):
        scenarios, tasks, requirements = semantic.scoped_objects(before, context.request.scope)
        explicit_scenarios = {key for scope in context.request.scope if scope.kind == "scenario" for key in scope.object_ids}
        if plan.axis_assessments != before.axis_assessments:
            raise ValueError("Axis changes require explicit whole-plan scope")
        for name, allowed in (("scenarios", explicit_scenarios), ("tasks", tasks), ("requirements", requirements),
                              ("acceptances", requirements), ("exclusions", tasks)):
            old, new = getattr(before, name), getattr(plan, name)
            for key in old.keys() | new.keys():
                if old.get(key) != new.get(key) and key not in allowed:
                    value = new.get(key)
                    related = key not in old and name == "tasks" and value and value.scenario_id in explicit_scenarios
                    related = related or (key not in old and name == "requirements" and value and (
                        bool(value.affected_scenario_ids) and set(value.affected_scenario_ids) <= explicit_scenarios
                        or bool(value.affected_task_ids) and set(value.affected_task_ids) <= tasks))
                    if not related:
                        raise ValueError("Draft change exceeds the delegated scope")
    plan.revision += 1
    semantic.validate_plan(plan, context.reader, task_data=context.task_data)
    path = _save(plan, context)
    return Command(
        update={
            "draft_path": path,
            "messages": [
                ToolMessage(
                    content=json.dumps(semantic.plan_summary(plan)),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _resolve_result(response, state, context):
    if state.get("active_request_id") != context.invocation_id:
        raise ValueError("Planner result belongs to another request")
    submitted = {
        item["artifact_id"]: ArtifactRef.model_validate(item) for item in state.get("submitted", [])
    }
    if len(response.artifact_ids) != len(set(response.artifact_ids)):
        raise ValueError("Duplicate final artifact IDs")
    if set(response.artifact_ids) - submitted.keys():
        raise ValueError("Final response may only select artifacts submitted in this request")
    artifacts = [submitted[key] for key in response.artifact_ids]
    for reference in artifacts:
        resolve_artifact(context.directory, reference)
    plan = _load(state, context)
    published = [ref for ref in artifacts if ref.kind == "task_plan"]
    if response.outcome == "completed" and not published:
        raise ValueError("Publish the current Task Plan before reporting completed")
    if published and (
        len(published) != 1
        or TaskPlan.model_validate(read_json_artifact(context.directory, published[0])) != plan
    ):
        raise ValueError("Select exactly the publication matching the current draft")
    _verify_acceptances(plan, context)
    pending = semantic.pending_requirements(plan)
    if response.outcome == "completed":
        semantic.check_operation_complete(plan, context.request.operation, context.request.scope, task_data=context.task_data)
    needs = []
    if published:
        for requirement_id in sorted(pending):
            requirement = plan.requirements[requirement_id]
            ref = next(
                (
                    item
                    for item in submitted.values()
                    if item.kind == "data_requirement"
                    and read_json_artifact(context.directory, item)
                    == requirement.model_dump(mode="json")
                ),
                None,
            )
            if ref is None:
                raise ValueError("Published plan is missing a registered DataRequirement")
            if ref not in artifacts:
                artifacts.append(ref)
            needs.append(
                WorkNeed(
                    need_id=requirement_id, kind="work", summary=requirement.purpose, detail_ref=ref
                )
            )
    if response.user_question:
        needs.append(WorkNeed(need_id="planner-user", kind="user", summary=response.user_question))
    return WorkResult(
        request_id=context.invocation_id,
        outcome=response.outcome,
        summary=response.summary,
        artifacts=artifacts,
        issues=response.issues,
        needs=needs,
    )


def _verify_acceptances(plan, context):
    bound = {ref.artifact_id: ref for ref in context.request.inputs}
    for requirement_id in semantic.active_requirements(plan) & plan.acceptances.keys():
        accepted = plan.acceptances[requirement_id]
        ref = bound.get(accepted.artifact_id)
        if ref is None or ref.digest != accepted.digest or ref.kind != "task_data":
            raise ValueError("Bind the exact accepted TaskData before publishing")
        _check_task_data(plan, context, ref.artifact_id, requirement_id)
    for identifier, accepted in plan.task_data_acceptances.items():
        if any(
            task.task_data
            and task.task_data.artifact_id == identifier
            and key not in plan.exclusions
            for key, task in plan.tasks.items()
        ):
            reader = _check_task_data(plan, context, identifier)
            if reader.content_digest != accepted.digest:
                raise ValueError("Accepted TaskData changed")


def _check_task_data(plan, context, identifier, requirement_id=None):
    reader = context.task_data[identifier]
    manifest = reader.manifest
    if manifest.dataset != plan.dataset:
        raise ValueError("TaskData binds a different Dataset")
    if isinstance(context.reader, ProfileReader) and (
        manifest.profile is None
        or manifest.profile.artifact_id != context.dataset.artifact_id
        or manifest.profile.digest != context.dataset.digest
    ):
        raise ValueError("TaskData must bind the current Profile")
    if set(manifest.scenario_ids) - plan.scenarios.keys():
        raise ValueError("TaskData references an unknown Scenario")
    if requirement_id is not None:
        requirement = plan.requirements[requirement_id]
        if requirement_id not in manifest.requirements or requirement.backend_id not in {
            None,
            manifest.backend_id,
        }:
            raise ValueError("TaskData does not answer this data requirement/backend")
        if set(requirement.affected_scenario_ids) - set(manifest.scenario_ids):
            raise ValueError("TaskData is missing required Scenarios")
        from medbenchagent.data.task_data import validate_preparation
        product = manifest.requirements[requirement_id]
        ref = next((r for r in context.request.inputs if r.artifact_id == product.artifact_id), None)
        if ref is None or ref.digest != product.digest or ref.kind != "data_requirement":
            raise ValueError("Bind the exact requirement used to prepare this TaskData")
        if read_json_artifact(context.directory, ref) != requirement.model_dump(mode="json"):
            raise ValueError("TaskData was prepared for a different requirement revision")
        validate_preparation(reader, requirement)
        if set(requirement.case_ids) - {case.case_id for case in reader.cases}:
            raise ValueError("TaskData is missing required Cases")
    reader.verify_integrity()
    return reader


def _recover_publications(context):
    """Recover only validated immutable submissions owned by the interrupted request."""
    output = context.directory / "artifacts" / context.invocation_id
    recovered = []
    for folder, kind, schema in (
        ("plans", "task_plan", TaskPlan),
        ("requirements", "data_requirement", DataRequirement),
    ):
        for path in sorted((output / folder).glob("*.json")):
            value = schema.model_validate_json(path.read_bytes())
            if path.stem != _product_digest(value):
                raise ValueError("Published content does not match its immutable submission name")
            if isinstance(value, TaskPlan):
                semantic.validate_plan(
                    value, context.reader, publish=True, task_data=context.task_data
                )
                _verify_acceptances(value, context)
            recovered.append(
                register_file(
                    context.directory,
                    context.request,
                    path.relative_to(context.directory).as_posix(),
                    kind=kind,
                )
            )
    return recovered
