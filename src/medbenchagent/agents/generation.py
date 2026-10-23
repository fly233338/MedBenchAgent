"""Native Generation agent exposing actual prepared-data backends and candidate operations."""

import hashlib
import json

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool, ToolRuntime

from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact, register_file
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.data.task_data import TaskDataReader
from medbenchagent.planning.contracts import TaskPlan, DataRequirement
from medbenchagent.generation.backends import backend_catalog
from medbenchagent.generation.contracts import (
    GenerationSpec,
    GenerationBatch,
    Allocation,
    validate_generation_spec,
)
from medbenchagent.generation.execution import (
    load_execution,
    selected_cases,
    allocate_counts,
    generate_batch,
    finalize_candidate,
    CandidateReader,
)
from medbenchagent.runtime.native import OperationTools, chat_model, loop_middleware
from .handoff import (
    WorkState,
    WorkContext,
    WorkResponse,
    WorkInstructions,
    binding,
    bound_path,
    get_ref,
    inspect_artifact,
    invoke_work,
    publish_value,
    tool_result,
)


class GenerationState(WorkState):
    quarantined: dict[str, str]


def generation_tools():
    @tool
    def inspect_generation_product(
        artifact_id: str,
        section: str,
        runtime: ToolRuntime[WorkContext],
        offset: int = 0,
        limit: int = 5,
        text_offset: int = 0,
    ) -> dict:
        """Read real items, case failures or quarantine records from a bound batch/candidate."""
        if offset < 0 or not 1 <= limit <= runtime.context.settings.page_size or text_offset < 0:
            raise ValueError("Invalid generation page")
        ref = get_ref(runtime, artifact_id)
        if ref.kind == "generation_candidate":
            candidate = CandidateReader(resolve_artifact(runtime.context.directory, ref))
            sections = {
                "items": [item.model_dump(mode="json") for item in candidate.items],
                "failures": [failure.model_dump() for failure in candidate.manifest.failures],
                "quarantine": candidate.quarantine,
            }
        elif ref.kind == "generation_batch":
            batch = GenerationBatch.model_validate(
                read_json_artifact(runtime.context.directory, ref)
            )
            sections = {
                "items": [item.model_dump(mode="json") for item in batch.items],
                "failures": [failure.model_dump() for failure in batch.failures],
            }
        else:
            raise ValueError("Select a generation batch or candidate")
        rows = sections[section]
        text = json.dumps(rows[offset : offset + limit], ensure_ascii=False)
        end = text_offset + runtime.context.settings.preview_chars
        return {
            "text": text[text_offset:end],
            "total": len(rows),
            "next_text_offset": end if end < len(text) else None,
            "next_offset": offset + limit if offset + limit < len(rows) else None,
        }

    @tool
    def list_generation_backends() -> list[dict]:
        """List implemented task-output backends, supported families and required reference shapes."""
        return backend_catalog()

    @tool
    def inspect_task_data(
        artifact_id: str,
        runtime: ToolRuntime[WorkContext],
        offset: int = 0,
        limit: int = 5,
        text_offset: int = 0,
    ) -> dict:
        """Read a bounded page of prepared Cases and their backend/scenario coverage."""
        if offset < 0 or not 1 <= limit <= runtime.context.settings.page_size or text_offset < 0:
            raise ValueError("Invalid Case page or text offset")
        reader = TaskDataReader(
            resolve_artifact(runtime.context.directory, get_ref(runtime, artifact_id, "task_data"))
        )
        text = json.dumps(
            [case.model_dump(mode="json") for case in reader.cases[offset : offset + limit]],
            ensure_ascii=False,
        )
        end = text_offset + runtime.context.settings.preview_chars
        return {
            "backend_id": reader.manifest.backend_id,
            "case_count": len(reader.cases),
            "scenario_counts": reader.manifest.scenario_counts,
            "text": text[text_offset:end],
            "next_text_offset": end if end < len(text) else None,
            "next_offset": offset + limit if offset + limit < len(reader.cases) else None,
        }

    @tool
    def prepare_task_execution(
        task_plan_id: str,
        task_id: str,
        task_data_id: str,
        sample_count: int,
        seed: int,
        runtime: ToolRuntime[WorkContext],
        case_ids: list[str] | None = None,
    ):
        """Validate a ready Leaf against accepted TaskData and publish its execution binding, without redesigning it."""
        plan_ref = get_ref(runtime, task_plan_id, "task_plan")
        data_ref = get_ref(runtime, task_data_id, "task_data")
        plan = TaskPlan.model_validate(read_json_artifact(runtime.context.directory, plan_ref))
        from medbenchagent.planning.semantic import scoped_objects
        scopes = [s for s in runtime.context.request.scope if s.artifact_id == task_plan_id]
        if not scopes or task_id not in scoped_objects(plan, scopes)[1]:
            raise ValueError("Task is outside the delegated scope")
        reader = TaskDataReader(resolve_artifact(runtime.context.directory, data_ref))
        spec = GenerationSpec(
            task_plan=binding(plan_ref),
            task_id=task_id,
            task_revision=plan.tasks[task_id].revision,
            task_data=binding(data_ref),
            case_ids=case_ids or [],
            sample_count=sample_count,
            seed=seed,
        )
        validate_generation_spec(
            spec,
            plan=plan,
            plan_artifact_id=plan_ref.artifact_id,
            plan_digest=plan_ref.digest,
            reader=reader,
        )
        ref = publish_value(runtime, "generation_spec", spec)
        return tool_result(
            runtime,
            {
                "artifact_id": ref.artifact_id,
                "eligible_count": len(selected_cases(spec, plan.tasks[task_id], reader)),
            },
            artifacts=[ref],
        )

    @tool
    def allocate_generation_budget(
        spec_ids: list[str], requested_count: int, runtime: ToolRuntime[WorkContext]
    ):
        """Allocate the unchanged total target over actual selected Case capacities; shortfall remains visible."""
        capacities = {}
        for identifier in spec_ids:
            ref = get_ref(runtime, identifier, "generation_spec")
            _, spec, plan, reader = load_execution(
                binding(ref), lambda p, k: bound_path(runtime, p, k)
            )
            if spec.task_id in capacities:
                raise ValueError("Choose only one specification per task")
            capacities[spec.task_id] = len(selected_cases(spec, plan.tasks[spec.task_id], reader))
        goals = [ref for ref in runtime.context.request.inputs if ref.kind == "benchmark_request"]
        if len(goals) != 1:
            raise ValueError("Bind exactly one benchmark_request with the user quantity target")
        if (
            read_json_artifact(runtime.context.directory, goals[0])["sample_count"]
            != requested_count
        ):
            raise ValueError("Allocation cannot reduce the bound user sample target")
        allocation = allocate_counts(capacities, requested_count, binding(goals[0]))
        ref = publish_value(runtime, "allocation", allocation)
        return tool_result(
            runtime, {"artifact_id": ref.artifact_id, **allocation.model_dump()}, artifacts=[ref]
        )

    @tool
    def generate_task_batch(
        spec_id: str, allocation_id: str, offset: int, limit: int, runtime: ToolRuntime[WorkContext]
    ):
        """Execute an allocated Case slice using the Leaf's backend; report case failures without fabricating answers."""
        if offset < 0 or not 1 <= limit <= runtime.context.settings.page_size:
            raise ValueError("Invalid batch range")
        ref = get_ref(runtime, spec_id, "generation_spec")
        _, spec, plan, reader = load_execution(binding(ref), lambda p, k: bound_path(runtime, p, k))
        allocation = Allocation.model_validate(
            read_json_artifact(
                runtime.context.directory, get_ref(runtime, allocation_id, "allocation")
            )
        )
        selected = selected_cases(spec, plan.tasks[spec.task_id], reader)[
            : allocation.counts[spec.task_id]
        ]
        batch = generate_batch(
            spec, binding(ref), plan.tasks[spec.task_id], reader, selected[offset : offset + limit]
        )
        output = publish_value(runtime, "generation_batch", batch)
        return tool_result(
            runtime,
            {
                "artifact_id": output.artifact_id,
                "generated": len(batch.items),
                "failures": [row.model_dump() for row in batch.failures],
                "next_offset": offset + limit if offset + limit < len(selected) else None,
            },
            artifacts=[output],
        )

    @tool
    def quarantine_generated_items(
        batch_ids: list[str], item_ids: list[str], reason: str, runtime: ToolRuntime[WorkContext]
    ):
        """Exclude selected generated items with an explicit reason; do not alter task definitions or references."""
        known = set()
        for identifier in batch_ids:
            batch = GenerationBatch.model_validate(
                read_json_artifact(
                    runtime.context.directory, get_ref(runtime, identifier, "generation_batch")
                )
            )
            known.update(item.item_id for item in batch.items)
        if not reason.strip() or set(item_ids) - known:
            raise ValueError("Quarantine requires known items and a reason")
        selected = {**runtime.state.get("quarantined", {}), **dict.fromkeys(item_ids, reason)}
        return tool_result(
            runtime, {"quarantined": item_ids, "reason": reason}, updates={"quarantined": selected}
        )

    @tool
    def propose_data_requirement(requirement: DataRequirement, runtime: ToolRuntime[WorkContext]):
        """Publish a concrete execution data need for Supervisor coordination; do not call another agent."""
        if requirement.source_role != "generation":
            raise ValueError("Generation requirements must declare source_role=generation")
        plans = [
            TaskPlan.model_validate(read_json_artifact(runtime.context.directory, ref))
            for ref in runtime.context.request.inputs
            if ref.kind == "task_plan"
        ]
        matching = [plan for plan in plans if plan.dataset == requirement.dataset]
        if not any(
            not (set(requirement.affected_task_ids) - plan.tasks.keys())
            and not (set(requirement.affected_scenario_ids) - plan.scenarios.keys())
            for plan in matching
        ):
            raise ValueError(
                "DataRequirement must bind the delegated TaskPlan and its actual tasks/Scenarios"
            )
        ref = publish_value(runtime, "data_requirement", requirement)
        return tool_result(
            runtime,
            {"artifact_id": ref.artifact_id, "requirement_id": requirement.requirement_id},
            artifacts=[ref],
        )

    @tool
    def finalize_generation(
        spec_ids: list[str],
        batch_ids: list[str],
        allocation_id: str,
        runtime: ToolRuntime[WorkContext],
    ):
        """Publish a verified immutable candidate and truthful counts from explicitly selected batches."""
        specs, batches = [], []
        for identifier in spec_ids:
            ref = get_ref(runtime, identifier, "generation_spec")
            actual, spec, _, _ = load_execution(
                binding(ref), lambda p, k: bound_path(runtime, p, k)
            )
            specs.append((actual, spec))
        for identifier in batch_ids:
            ref = get_ref(runtime, identifier, "generation_batch")
            batches.append(
                (
                    binding(ref),
                    GenerationBatch.model_validate(
                        read_json_artifact(runtime.context.directory, ref)
                    ),
                )
            )
        allocation_ref = get_ref(runtime, allocation_id, "allocation")
        allocation = Allocation.model_validate(
            read_json_artifact(runtime.context.directory, allocation_ref)
        )
        known = {item.item_id for _, batch in batches for item in batch.items}
        quarantined = {
            key: reason
            for key, reason in runtime.state.get("quarantined", {}).items()
            if key in known
        }
        identity = hashlib.sha256(
            json.dumps(
                {
                    "specs": spec_ids,
                    "batches": batch_ids,
                    "allocation": allocation_ref.model_dump(),
                    "quarantine": quarantined,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        destination = (
            runtime.context.directory
            / "artifacts"
            / runtime.context.invocation_id
            / "candidates"
            / identity
        )
        path = finalize_candidate(
            destination,
            batches=batches,
            specs=specs,
            allocation=allocation,
            quarantined=quarantined,
        )
        ref = register_file(
            runtime.context.directory,
            runtime.context.request,
            path.relative_to(runtime.context.directory).as_posix(),
            kind="generation_candidate",
        )
        return tool_result(
            runtime,
            {
                "artifact_id": ref.artifact_id,
                "counts": json.loads(path.read_text(encoding="utf-8"))["counts"],
            },
            artifacts=[ref],
        )

    return [
        inspect_artifact,
        inspect_generation_product,
        list_generation_backends,
        inspect_task_data,
        prepare_task_execution,
        allocate_generation_budget,
        generate_task_batch,
        quarantine_generated_items,
        propose_data_requirement,
        finalize_generation,
    ]


def build_generation_spec(settings, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    tools = generation_tools()
    agent = create_agent(
        model,
        tools=tools,
        response_format=ToolStrategy(WorkResponse),
        state_schema=GenerationState,
        context_schema=WorkContext,
        checkpointer=True,
        name="generation_agent",
        middleware=[
            WorkInstructions(GenerationState, {"quarantined": {}}),
            OperationTools({'generate': None}),
            *loop_middleware(
                model,
                settings.loop,
                settings.context_chars,
                write_tools=[t.name for t in tools if not t.name.startswith(("inspect_", "list_"))],
            ),
        ],
    )

    def node(state, config, runtime):
        return invoke_work(
            agent,
            WorkContext(
                runtime.context.directory,
                WorkRequest.model_validate(state["request"]),
                settings,
                instructions,
                "generation",
            ),
            config,
        )

    return AgentSpec(
        "Prepare execution, allocate counts and assemble questions from ready Leaf definitions and accepted TaskData. "
        "Supports VQA, reports, grounding and detection; returns immutable batches/candidates and data or task issues. "
        "Bind TaskPlans, TaskData and any referenced specs/batches explicitly.",
        node, operations=('generate',),
    )
