"""Task Planning tools."""

import json
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from medbenchagent.compilation.artifacts import (
    read_json_artifact,
    write_json_artifact,
)
from medbenchagent.data.task_data import ProfileReader
from medbenchagent.planning.contracts import (
    Axis,
    AxisAssessment,
    DataRequirement,
    RequirementAcceptance,
    SemanticTaskSpec,
    TaskExclusion,
    PlanningEvidenceRef,
)
from medbenchagent.planning import semantic

from .state import PlannerContext, _bounded, _check_task_data, _load, _product_digest, _update, _verify_acceptances


def planner_tools():
    @tool
    def inspect_profile(runtime: ToolRuntime[PlannerContext]) -> dict:
        """Read source Profile coverage, observations and unknowns; no fully normalized Dataset is required."""
        if not isinstance(runtime.context.reader, ProfileReader):
            raise ValueError("The current input is a canonical Dataset, not a Profile")
        return _bounded(
            runtime.context.reader.read_mapping(), runtime.context.settings.preview_chars
        )

    @tool
    def list_generation_backends() -> list[dict]:
        """Read actual backend families, prepared-data expectations and supported output contracts."""
        from medbenchagent.generation.backends import backend_catalog

        return backend_catalog()

    @tool
    def inspect_task_data(
        artifact_id: str,
        runtime: ToolRuntime[PlannerContext],
        case_id: str | None = None,
        offset: int = 0,
        limit: int = 10,
        text_offset: int = 0,
    ) -> dict:
        """Inspect bound prepared Cases, fields and coverage for a backend, with bounded pages."""
        reader = runtime.context.task_data[artifact_id]
        reader.verify_integrity()
        if text_offset < 0:
            raise ValueError("text_offset must be nonnegative")
        if case_id is not None:
            text = reader.get_case(case_id).model_dump_json()
            end = text_offset + runtime.context.settings.preview_chars
            return {"artifact_id": artifact_id, "digest": reader.content_digest,
                    "case_id": case_id, "text": text[text_offset:end],
                    "next_text_offset": end if end < len(text) else None}
        rows = [
            {
                "case_id": case.case_id,
                "scenario_ids": case.scenario_ids,
                "fields": [
                    {"name": f.name, "type": f.type, "input_visibility": f.input_visibility}
                    for f in case.fields
                ],
            }
            for case in reader.cases
        ]
        result = semantic.page(
            reader,
            rows,
            offset,
            limit,
            scope={"task_data": artifact_id},
            ceiling=runtime.context.settings.page_size,
        )
        result["backend_id"] = reader.manifest.backend_id
        return _bounded(result, runtime.context.settings.preview_chars)

    @tool
    def inspect_dataset(runtime: ToolRuntime[PlannerContext]) -> dict:
        """Read complete deterministic counts and a bounded evidence-name catalog for this snapshot."""
        reader = runtime.context.reader
        if isinstance(reader, ProfileReader):
            profile = reader.profile
            reader.verify_integrity()
            return {
                "dataset": profile.dataset.model_dump(),
                "coverage": profile.coverage,
                "inspected_scope": profile.inspected_scope,
                "case_count": None,
                "inspected_sample_count": len(profile.samples),
                "unknowns": profile.unknowns,
                "observations": [
                    o.model_dump(mode="json")
                    for o in profile.observations[: runtime.context.settings.page_size]
                ],
                "truncated": len(profile.observations) > runtime.context.settings.page_size,
            }
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
        evidence_refs: list[PlanningEvidenceRef],
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
    def set_scenario_backends(
        scenario_id: str,
        backend_ids: list[str],
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Select implemented backends for a Scenario; backend selection does not add an Axis."""

        def apply(plan):
            if len(backend_ids) != len(set(backend_ids)):
                raise ValueError("Duplicate backend IDs")
            plan.scenarios[scenario_id].backend_ids = backend_ids

        return _update(runtime, base_revision, apply)

    @tool
    def accept_task_data(
        artifact_id: str,
        reason: str,
        evidence_refs: list[PlanningEvidenceRef],
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Accept inspected prepared TaskData as a basis for Leaf design, with your evidence and reason."""

        def apply(plan):
            reader = _check_task_data(plan, runtime.context, artifact_id)
            plan.task_data_acceptances[artifact_id] = RequirementAcceptance(
                artifact_id=artifact_id,
                digest=reader.content_digest,
                reason=reason,
                evidence_refs=evidence_refs,
            )

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
        evidence_refs: list[PlanningEvidenceRef],
        base_revision: int,
        runtime: ToolRuntime[PlannerContext],
    ) -> Command:
        """Accept the exact TaskData against the current requirement; cite fields in that TaskData."""

        def apply(plan):
            reader = _check_task_data(plan, runtime.context, artifact_id, requirement_id)
            acceptance = RequirementAcceptance(
                artifact_id=artifact_id, digest=reader.content_digest,
                reason=reason, evidence_refs=evidence_refs,
            )
            plan.acceptances[requirement_id] = acceptance
            plan.task_data_acceptances[artifact_id] = acceptance

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
        semantic.validate_plan(plan, context.reader, publish=True, task_data=context.task_data)
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
        inspect_profile,
        list_generation_backends,
        inspect_task_data,
        set_scenario_backends,
        accept_task_data,
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
