"""Independent native review over explicitly bound plans, candidates and evidence."""

import json

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool, ToolRuntime

from medbenchagent.compilation.artifacts import read_json_artifact, resolve_artifact
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.generation.contracts import AuditReport, ReviewFinding, ReviewReport
from medbenchagent.generation.execution import CandidateReader, load_execution
from medbenchagent.audit.generation import audit_candidate
from medbenchagent.planning.contracts import TaskPlan
from medbenchagent.data.task_data import TaskDataReader
from medbenchagent.runtime.native import chat_model, loop_middleware
from .handoff import (
    WorkContext,
    WorkState,
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


def review_tools():
    @tool
    def inspect_task_case(
        task_data_id: str, case_id: str, runtime: ToolRuntime[WorkContext], text_offset: int = 0
    ) -> dict:
        """Read the actual prepared evidence for a candidate Case; reference data is for review, not model input."""
        if text_offset < 0:
            raise ValueError("Invalid text offset")
        ref = get_ref(runtime, task_data_id, "task_data")
        reader = TaskDataReader(resolve_artifact(runtime.context.directory, ref))
        text = reader.get_case(case_id).model_dump_json()
        end = text_offset + runtime.context.settings.preview_chars
        return {"text": text[text_offset:end], "next_text_offset": end if end < len(text) else None}

    @tool
    def inspect_candidate(
        artifact_id: str,
        runtime: ToolRuntime[WorkContext],
        offset: int = 0,
        limit: int = 5,
        text_offset: int = 0,
    ) -> dict:
        """Read candidate counts and real question samples, with explicit pagination and text continuation."""
        if offset < 0 or not 1 <= limit <= runtime.context.settings.page_size or text_offset < 0:
            raise ValueError("Invalid sample range")
        candidate = CandidateReader(
            resolve_artifact(
                runtime.context.directory, get_ref(runtime, artifact_id, "generation_candidate")
            )
        )
        text = json.dumps(
            [item.model_dump(mode="json") for item in candidate.items[offset : offset + limit]],
            ensure_ascii=False,
        )
        end = text_offset + runtime.context.settings.preview_chars
        return {
            "counts": candidate.manifest.counts,
            "text": text[text_offset:end],
            "next_text_offset": end if end < len(text) else None,
            "next_offset": offset + limit if offset + limit < len(candidate.items) else None,
        }

    @tool
    def run_candidate_audit(candidate_id: str, runtime: ToolRuntime[WorkContext]):
        """Run complete deterministic candidate checks against exact Leaf, spec and TaskData bindings."""
        ref = get_ref(runtime, candidate_id, "generation_candidate")
        candidate = CandidateReader(resolve_artifact(runtime.context.directory, ref))
        target_path = bound_path(runtime, candidate.manifest.allocation.target, "benchmark_request")
        if (
            json.loads(target_path.read_text(encoding="utf-8"))["sample_count"]
            != candidate.manifest.allocation.requested
        ):
            raise ValueError("Candidate quantity differs from its bound user target")
        executions = {
            spec.artifact_id: load_execution(spec, lambda p, k: bound_path(runtime, p, k))
            for spec in candidate.manifest.specs
        }
        report = audit_candidate(candidate, binding(ref), executions)
        output = publish_value(runtime, "audit_report", report)
        return tool_result(
            runtime,
            {"artifact_id": output.artifact_id, **report.model_dump(mode="json")},
            artifacts=[output],
        )

    @tool
    def finalize_review(
        target_id: str,
        findings: list[ReviewFinding],
        checked_task_ids: list[str],
        checked_scenario_ids: list[str],
        checked_item_ids: list[str],
        unchecked_scope: list[str],
        approved: bool,
        runtime: ToolRuntime[WorkContext],
        audit_id: str | None = None,
    ):
        """Submit semantic findings and actual review coverage. Candidate approval requires a matching complete Audit."""
        target = get_ref(runtime, target_id)
        sampled = False
        if target.kind == "generation_candidate":
            candidate = CandidateReader(resolve_artifact(runtime.context.directory, target))
            task_ids = set(candidate.manifest.allocation.counts)
            items = {item.item_id: item for item in candidate.items}
            scenarios = {item.scenario_id for item in candidate.items}
            if set(checked_item_ids) - items.keys():
                raise ValueError("Review references an item outside this candidate")
            if audit_id is None:
                raise ValueError("Candidate review requires a full Audit")
            audit_ref = get_ref(runtime, audit_id, "audit_report")
            audit = AuditReport.model_validate(
                read_json_artifact(runtime.context.directory, audit_ref)
            )
            if audit.target != binding(target) or audit.checked_count != len(items):
                raise ValueError("Audit does not cover the current candidate")
            sampled_tasks = {items[key].leaf_task_id for key in checked_item_ids}
            nonempty_tasks = {item.leaf_task_id for item in candidate.items}
            if approved and (not audit.passed or nonempty_tasks - sampled_tasks):
                raise ValueError(
                    "Approval needs a passing Audit and a semantic sample for each generated task"
                )
            sampled = len(checked_item_ids) < len(items)
        elif target.kind == "task_plan":
            plan = TaskPlan.model_validate(read_json_artifact(runtime.context.directory, target))
            task_ids, scenarios, items = set(plan.tasks), set(plan.scenarios), {}
            audit_ref = None
            if checked_item_ids or audit_id is not None:
                raise ValueError("A Task Plan review has no candidate items or candidate Audit")
        else:
            raise ValueError("Review targets a task_plan or generation_candidate")
        if set(checked_task_ids) - task_ids or set(checked_scenario_ids) - scenarios:
            raise ValueError("Review coverage contains unknown tasks or Scenarios")
        if approved and (
            set(checked_task_ids) != task_ids
            or set(checked_scenario_ids) != scenarios
            or unchecked_scope
        ):
            raise ValueError(
                "Approval requires complete task/Scenario coverage and no outstanding scope"
            )
        known_objects = {target.artifact_id, *task_ids, *scenarios, *items}
        for finding in findings:
            if finding.object_id not in known_objects:
                raise ValueError("Finding references an unknown review object")
            for identifier in finding.evidence_artifact_ids:
                get_ref(runtime, identifier)
        if approved and any(finding.blocking for finding in findings):
            raise ValueError("Blocking findings cannot be approved")
        report = ReviewReport(
            target=binding(target),
            audit=binding(audit_ref) if audit_ref else None,
            findings=findings,
            checked_task_ids=checked_task_ids,
            checked_scenario_ids=checked_scenario_ids,
            checked_item_ids=checked_item_ids,
            unchecked_scope=unchecked_scope,
            approved=approved,
            semantic_sampled=sampled,
        )
        output = publish_value(runtime, "review_report", report)
        return tool_result(
            runtime,
            {"artifact_id": output.artifact_id, "approved": approved, "semantic_sampled": sampled},
            artifacts=[output],
        )

    return [
        inspect_artifact,
        inspect_task_case,
        inspect_candidate,
        run_candidate_audit,
        finalize_review,
    ]


def build_review_spec(settings, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    agent = create_agent(
        model,
        tools=review_tools(),
        response_format=ToolStrategy(WorkResponse),
        state_schema=WorkState,
        context_schema=WorkContext,
        checkpointer=True,
        name="review_agent",
        middleware=[
            WorkInstructions(),
            *loop_middleware(
                model,
                settings.loop,
                settings.context_chars,
                write_tools=("run_candidate_audit", "finalize_review"),
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
                "review",
            ),
            config,
        )

    return AgentSpec(
        "Independent semantic review of TaskPlans or generation candidates. Read-only on targets; writes findings, "
        "coverage and deterministic Audit/Review reports bound to exact target content. Candidate review requires "
        "its specs, TaskPlans and TaskData inputs. Return issues to Supervisor without modifying another agent's products.",
        node,
    )
