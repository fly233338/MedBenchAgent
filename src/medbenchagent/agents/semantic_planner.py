"""Native semantic Planner; tool choices belong to the model, operations to planning."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import Field

from medbenchagent.compilation.artifacts import (
    read_json_artifact,
    resolve_artifact,
    write_json_artifact,
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
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.data.models import EvidenceRef
from medbenchagent.planning.contracts import (
    Axis,
    AxisAssessment,
    DataRequirement,
    RequirementAcceptance,
    SemanticTaskSpec,
    TaskExclusion,
    TaskPlan,
)
from medbenchagent.planning import semantic
from medbenchagent.runtime.native import AGENT_LIMIT_ERRORS, chat_model, loop_middleware, role_lock


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
    reader: FileDatasetReader
    dataset: ArtifactRef
    settings: object
    instructions: str
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
    operation(plan)
    plan.revision += 1
    semantic.validate_plan(plan, context.reader)
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
    if pending and response.outcome == "completed":
        raise ValueError("Unaccepted data requirements remain; report needs_input or progress")
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
        if ref is None or ref.digest != accepted.digest or ref.kind != "derived_view":
            raise ValueError(
                "Bind the accepted derived_view artifact to this request before publishing or completing"
            )
        value = read_json_artifact(context.directory, ref)
        if (
            value.get("dataset") != plan.dataset.model_dump()
            or value.get("requirement_id") != requirement_id
        ):
            raise ValueError("Accepted derived_view does not match this requirement and Dataset")


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
                semantic.validate_plan(value, context.reader, publish=True)
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


class PlannerContextMiddleware(AgentMiddleware):
    state_schema = PlannerState

    def before_agent(self, state, runtime):
        context = runtime.context
        if state.get("active_request_id") == context.invocation_id:
            return None
        plans = [ref for ref in context.request.inputs if ref.kind == "task_plan"]
        if len(plans) > 1:
            raise ValueError("Bind exactly one Task Plan to revise")
        if plans:
            plan = TaskPlan.model_validate(read_json_artifact(context.directory, plans[0]))
        elif state.get("draft_path"):
            plan = _load(state, context)
            if plan.dataset_digest != context.dataset.digest:
                plan = None
        else:
            plan = None
        if plan is None:
            plan = TaskPlan(
                plan_id=f"{context.compilation_id}-plan-{context.dataset.digest[:12]}",
                dataset=semantic.dataset_ref(context.reader),
                dataset_digest=context.dataset.digest,
            )
        if plan.dataset_digest != context.dataset.digest:
            raise ValueError(
                "Task Plan binds a different dataset artifact; begin a new plan for new evidence"
            )
        semantic.validate_plan(plan, context.reader)
        return {
            "active_request_id": context.invocation_id,
            "draft_path": _save(plan, context),
            "submitted": [],
            "structured_response": None,
        }

    def wrap_model_call(self, request, handler):
        context = request.runtime.context
        plan = _load(request.state, context)
        current = {
            "request": context.request.model_dump(mode="json"),
            "dataset": semantic.dataset_ref(context.reader),
            "plan": semantic.plan_summary(plan),
            "submitted": [
                {"artifact_id": item["artifact_id"], "kind": item["kind"]}
                for item in request.state.get("submitted", [])
            ],
            "query_limits": {
                "page_size": context.settings.page_size,
                "preview_chars": context.settings.preview_chars,
            },
        }
        response = handler(
            request.override(
                system_message=SystemMessage(
                    content=context.instructions
                    + "\nCurrent authoritative view:\n"
                    + json.dumps(current, ensure_ascii=False)
                )
            )
        )
        if response.structured_response is not None:
            try:
                _resolve_result(response.structured_response, request.state, context)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                message = response.result[0]
                return ModelResponse(
                    result=[
                        message,
                        *[
                            ToolMessage(content=str(exc), tool_call_id=call["id"], status="error")
                            for call in message.tool_calls
                        ],
                    ]
                )
        return response


def planner_tools():
    @tool
    def inspect_dataset(runtime: ToolRuntime[PlannerContext]) -> dict:
        """Read complete deterministic counts and a bounded evidence-name catalog for this snapshot."""
        reader = runtime.context.reader
        catalog = semantic.evidence_catalog(reader)
        return {
            "dataset": semantic.dataset_ref(reader),
            "case_count": sum(1 for _ in semantic.cases(reader)),
            "evidence": catalog[: runtime.context.settings.page_size],
            "total_fields": len(catalog),
            "truncated": len(catalog) > runtime.context.settings.page_size,
        }

    @tool
    def read_evidence_catalog(
        runtime: ToolRuntime[PlannerContext], offset: int = 0, limit: int = 20
    ) -> dict:
        """Page through evidence names, counts, visibility and representative Evidence references."""
        return semantic.page(
            runtime.context.reader,
            semantic.evidence_catalog(runtime.context.reader),
            offset,
            limit,
            scope="all Cases",
            ceiling=runtime.context.settings.page_size,
        )

    @tool
    def read_field_values(
        name: str,
        runtime: ToolRuntime[PlannerContext],
        offset: int = 0,
        limit: int = 20,
        text_offset: int = 0,
    ) -> dict:
        """Read one field across Cases. Pagination and text continuation are explicit; unknown stays unknown."""
        if text_offset < 0:
            raise ValueError("text_offset must be nonnegative")
        context = runtime.context
        result = semantic.page(
            context.reader,
            semantic.field_values(context.reader, name),
            offset,
            limit,
            scope={"field": name},
            ceiling=context.settings.page_size,
        )
        for row in result["items"]:
            text = json.dumps(row.pop("value"), ensure_ascii=False)
            end = text_offset + max(1, context.settings.preview_chars // max(1, limit))
            row.update(
                value_text=text[text_offset:end],
                text_offset=text_offset,
                next_text_offset=end if end < len(text) else None,
                total_chars=len(text),
            )
        return result

    @tool
    def read_case(
        case_id: str, runtime: ToolRuntime[PlannerContext], offset: int = 0, limit: int = 10
    ) -> dict:
        """Read a Case's evidence page; use the evidence query for full values of truncated fields."""
        context = runtime.context
        case = context.reader.get_case(case_id)
        rows = [item.model_dump(mode="json") for item in case.evidence]
        result = semantic.page(
            context.reader,
            rows,
            offset,
            limit,
            scope={"case_id": case_id},
            ceiling=context.settings.page_size,
        )
        result["items"] = [
            _bounded(row, context.settings.preview_chars // max(1, limit))
            for row in result["items"]
        ]
        result["relations"] = _bounded(
            [r.model_dump(mode="json") for r in case.relations], context.settings.preview_chars
        )
        return result

    @tool
    def read_mapping(runtime: ToolRuntime[PlannerContext], text_offset: int = 0) -> dict:
        """Read the source semantics and Case mapping as bounded JSON text with continuation."""
        if text_offset < 0:
            raise ValueError("text_offset must be nonnegative")
        mapping = runtime.context.reader.read_mapping()
        if mapping is None:
            return {
                "status": "unknown",
                "reason": "No mapping was supplied",
                "dataset": semantic.dataset_ref(runtime.context.reader),
            }
        text = json.dumps(mapping, ensure_ascii=False)
        end = text_offset + runtime.context.settings.preview_chars
        return {
            "dataset": semantic.dataset_ref(runtime.context.reader),
            "text": text[text_offset:end],
            "total_chars": len(text),
            "next_text_offset": end if end < len(text) else None,
            "truncated": end < len(text),
        }

    @tool
    def inspect_task_plan(
        section: str,
        runtime: ToolRuntime[PlannerContext],
        offset: int = 0,
        limit: int = 20,
        object_id: str | None = None,
        text_offset: int = 0,
    ) -> dict:
        """Read draft sections: axis_assessments, scenarios, tasks, requirements, acceptances, exclusions."""
        plan = _load(runtime.state, runtime.context)
        if section not in {
            "axis_assessments",
            "scenarios",
            "tasks",
            "requirements",
            "acceptances",
            "exclusions",
        }:
            raise ValueError("Unknown Task Plan section")
        if text_offset < 0:
            raise ValueError("text_offset must be nonnegative")
        if object_id is not None:
            text = json.dumps(plan.model_dump(mode="json")[section][object_id], ensure_ascii=False)
            end = text_offset + runtime.context.settings.preview_chars
            return {
                "revision": plan.revision,
                "id": object_id,
                "text": text[text_offset:end],
                "next_text_offset": end if end < len(text) else None,
                "total_chars": len(text),
                "truncated": end < len(text),
            }
        rows = [
            {"id": key, "value": value}
            for key, value in plan.model_dump(mode="json")[section].items()
        ]
        result = semantic.page(
            runtime.context.reader,
            rows,
            offset,
            limit,
            scope={"section": section},
            ceiling=runtime.context.settings.page_size,
        )
        result["revision"] = plan.revision
        result["items"] = [
            {
                "id": row["id"],
                "value": _bounded(
                    row["value"], runtime.context.settings.preview_chars // max(1, limit)
                ),
            }
            for row in result["items"]
        ]
        return result

    @tool
    def inspect_input_artifact(
        artifact_id: str, runtime: ToolRuntime[PlannerContext], text_offset: int = 0
    ) -> dict:
        """Read a delegated report or derived-data artifact; only explicitly bound input IDs are accessible."""
        if text_offset < 0:
            raise ValueError("text_offset must be nonnegative")
        context = runtime.context
        ref = next(
            (item for item in context.request.inputs if item.artifact_id == artifact_id), None
        )
        if ref is None:
            raise ValueError("Artifact is not an input of this request")
        text = json.dumps(read_json_artifact(context.directory, ref), ensure_ascii=False)
        end = text_offset + context.settings.preview_chars
        return {
            "artifact_id": artifact_id,
            "digest": ref.digest,
            "text": text[text_offset:end],
            "total_chars": len(text),
            "next_text_offset": end if end < len(text) else None,
            "truncated": end < len(text),
        }

    @tool
    def record_axis_assessment(
        axis: Axis,
        assessments: list[AxisAssessment],
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Submit your supported/unsupported Axis judgments with evidence. Revising invalidates Scenario decisions."""

        def apply(plan):
            if not assessments:
                raise ValueError("Assess at least one Axis value")
            previous = {row.value: row for row in plan.axis_assessments.get(axis, [])}
            incoming = {row.value: row for row in assessments}
            changed = {
                key
                for key in previous.keys() | incoming.keys()
                if previous.get(key) != incoming.get(key)
            }
            affected = {
                key
                for key, scenario in plan.scenarios.items()
                if getattr(scenario, axis) in changed
            }
            if any(
                task_id not in plan.exclusions and task.scenario_id in affected
                for task_id, task in plan.tasks.items()
            ):
                raise ValueError(
                    "Exclude affected tasks before revising Axes; their semantics need reassessment"
                )
            plan.axis_assessments[axis] = assessments
            for key in affected:
                plan.scenarios[key].decision = "pending"
                plan.scenarios[key].reason = ""
                plan.scenarios[key].evidence_refs = []

        return _update(runtime, base_revision, apply)

    @tool
    def build_candidate_scenarios(
        base_revision: int, runtime: ToolRuntime[PlannerContext]
    ) -> Command:
        """Construct the Cartesian product of supported Axis values without making semantic judgments."""
        return _update(
            runtime,
            base_revision,
            lambda plan: semantic.build_scenarios(
                plan, max_scenarios=runtime.context.settings.max_scenarios
            ),
        )

    @tool
    def record_scenario_decision(
        scenario_id: str,
        keep: bool,
        reason: str,
        evidence_refs: list[EvidenceRef],
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Record your semantic screening judgment and its evidence; this does not assess backend readiness."""

        def apply(plan):
            scenario = plan.scenarios[scenario_id]
            scenario.decision = "keep" if keep else "exclude"
            scenario.reason, scenario.evidence_refs = reason, evidence_refs

        return _update(runtime, base_revision, apply)

    @tool
    def save_semantic_task(
        task: SemanticTaskSpec, base_revision: int, runtime: ToolRuntime[PlannerContext]
    ) -> Command:
        """Create or revise a semantic task. Revision must advance by one; bind input and reference evidence separately."""

        def apply(plan):
            old = plan.tasks.get(task.task_id)
            if task.revision != (old.revision + 1 if old else 1):
                raise ValueError("Task revision must advance by exactly one")
            plan.tasks[task.task_id] = task
            plan.exclusions.pop(task.task_id, None)

        return _update(runtime, base_revision, apply)

    @tool
    def propose_data_requirement(
        requirement: DataRequirement, base_revision: int, runtime: ToolRuntime[PlannerContext]
    ) -> Command:
        """Save the required data representation, affected tasks, evidence and acceptance conditions."""

        def apply(plan):
            if requirement.requirement_id in plan.acceptances:
                raise ValueError("Use a new requirement ID when changing an accepted requirement")
            plan.requirements[requirement.requirement_id] = requirement

        return _update(runtime, base_revision, apply)

    @tool
    def accept_data_requirement(
        requirement_id: str,
        artifact_id: str,
        reason: str,
        evidence_refs: list[EvidenceRef],
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Submit your acceptance of a delegated derived_view after inspecting its actual content and criteria."""

        def apply(plan):
            requirement = plan.requirements[requirement_id]
            ref = next(
                (
                    item
                    for item in runtime.context.request.inputs
                    if item.artifact_id == artifact_id and item.kind == "derived_view"
                ),
                None,
            )
            if ref is None:
                raise ValueError("Acceptance requires a bound derived_view input")
            value = read_json_artifact(runtime.context.directory, ref)
            if (
                value.get("dataset") != plan.dataset.model_dump()
                or value.get("requirement_id") != requirement_id
            ):
                raise ValueError("Derived view does not answer this requirement on this Dataset")
            rows = value.get("rows", [])
            case_ids = [row["case_id"] for row in rows]
            if len(case_ids) != len(set(case_ids)) or set(requirement.case_ids) - set(case_ids):
                raise ValueError("Derived view has duplicate or missing required Cases")
            for case_id in case_ids:
                runtime.context.reader.get_case(case_id)
            plan.acceptances[requirement_id] = RequirementAcceptance(
                artifact_id=artifact_id,
                digest=ref.digest,
                reason=reason,
                evidence_refs=evidence_refs,
            )

        return _update(runtime, base_revision, apply)

    @tool
    def exclude_semantic_task(
        task_id: str,
        exclusion: TaskExclusion,
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Explicitly exclude a task with its reason and evidence, retaining the task definition for inspection."""

        def apply(plan):
            if task_id not in plan.tasks:
                raise ValueError("Unknown task ID")
            plan.exclusions[task_id] = exclusion

        return _update(runtime, base_revision, apply)

    @tool
    def publish_task_plan(base_revision: int, runtime: ToolRuntime[PlannerContext]) -> Command:
        """Validate and publish the current immutable Task Plan and its DataRequirements; returns registered IDs."""
        context = runtime.context
        plan = _load(runtime.state, context)
        if plan.revision != base_revision:
            raise ValueError("Stale draft revision")
        semantic.validate_plan(plan, context.reader, publish=True)
        _verify_acceptances(plan, context)
        references = []
        for requirement in plan.requirements.values():
            digest = _product_digest(requirement)
            references.append(
                write_json_artifact(
                    context.directory,
                    context.request,
                    f"requirements/{digest}.json",
                    kind="data_requirement",
                    value=requirement.model_dump(mode="json"),
                )
            )
        digest = _product_digest(plan)
        references.append(
            write_json_artifact(
                context.directory,
                context.request,
                f"plans/{digest}.json",
                kind="task_plan",
                value=plan.model_dump(mode="json"),
            )
        )
        registered = {item["artifact_id"]: item for item in runtime.state.get("submitted", [])}
        registered.update({ref.artifact_id: ref.model_dump(mode="json") for ref in references})
        return Command(
            update={
                "submitted": list(registered.values()),
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            {
                                "revision": plan.revision,
                                "artifacts": [
                                    {"artifact_id": ref.artifact_id, "kind": ref.kind}
                                    for ref in references
                                ],
                            }
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return [
        inspect_dataset,
        read_evidence_catalog,
        read_field_values,
        read_case,
        read_mapping,
        inspect_task_plan,
        inspect_input_artifact,
        record_axis_assessment,
        build_candidate_scenarios,
        record_scenario_decision,
        save_semantic_task,
        propose_data_requirement,
        accept_data_requirement,
        exclude_semantic_task,
        publish_task_plan,
    ]


def build_planner_spec(settings, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    tools = planner_tools()
    writes = [
        item.name
        for item in tools
        if item.name.startswith(
            ("record_", "build_", "save_", "propose_", "accept_", "exclude_", "publish_")
        )
    ]
    agent = create_agent(
        model,
        tools=tools,
        response_format=ToolStrategy(PlannerResponse),
        state_schema=PlannerState,
        context_schema=PlannerContext,
        checkpointer=True,
        name="task_planner_agent",
        middleware=[
            PlannerContextMiddleware(),
            *loop_middleware(model, settings.loop, settings.context_chars, write_tools=writes),
        ],
    )

    def node(state, config, runtime):
        request = WorkRequest.model_validate(state["request"])
        directory = runtime.context.directory
        try:
            datasets = [ref for ref in request.inputs if ref.kind == "dataset"]
            if len(datasets) != 1:
                raise ValueError("Planner requires exactly one bound Dataset artifact")
            context = PlannerContext(
                directory,
                request,
                FileDatasetReader(resolve_artifact(directory, datasets[0])),
                datasets[0],
                settings,
                instructions,
            )
            with role_lock(directory, "task_planner"):
                result = agent.invoke(
                    {
                        "messages": [
                            HumanMessage(content=request.objective, id=request.request_id)
                        ],
                        "structured_response": None,
                    },
                    {**config, "recursion_limit": settings.loop.max_model_calls * 6 + 20},
                    context=context,
                )
            response = result.get("structured_response")
            if response is None:
                raise ValueError("Planner ended without a current structured result")
            output = _resolve_result(response, result, context)
        except (ValueError, KeyError, OSError) as exc:
            output = WorkResult(
                request_id=request.request_id,
                outcome="failed",
                summary="Planner work could not be accepted",
                issues=[WorkIssue(code="invalid_planning_work", message=str(exc))],
            )
        except AGENT_LIMIT_ERRORS as exc:
            output = WorkResult(
                request_id=request.request_id,
                outcome="failed",
                summary="Planner invocation budget exhausted",
                artifacts=_recover_publications(context),
                issues=[WorkIssue(code="agent_limit", message=str(exc))],
            )
        return {"result": output.model_dump(mode="json")}

    return AgentSpec(
        "Semantic task planning from a canonical Dataset: evidence queries, Axis/Scenario judgments, "
        "task revision, data requirements and formal Task Plan publication. Requires one dataset; "
        "may consume a selected task_plan and derived_view inputs. Does not generate benchmark items.",
        node,
    )
