"""Deterministic semantic planning operations over the canonical Case contract."""

import hashlib
import itertools
import json
from collections import Counter

from .contracts import SemanticScenario, TaskPlan


AXES = ("input_structure", "operation", "clinical_target")


def dataset_ref(reader):
    manifest = reader.get_manifest()
    return {"dataset_id": manifest.dataset_id, "revision": manifest.revision}


def cases(reader):
    offset = 0
    while True:
        page = reader.list_cases(offset, 100)
        yield from page.items
        if page.next_offset is None:
            break
        offset = page.next_offset


def page(reader, rows, offset, limit, *, scope, ceiling=20):
    if offset < 0 or limit < 1 or limit > ceiling or offset > len(rows):
        raise ValueError(f"Use offset within total and limit between 1 and {ceiling}")
    end = min(offset + limit, len(rows))
    return {
        "dataset": dataset_ref(reader),
        "scope": scope,
        "items": rows[offset:end],
        "total": len(rows),
        "total_known": True,
        "returned": end - offset,
        "next_offset": end if end < len(rows) else None,
        "truncated": end < len(rows),
        "status": "found" if rows else "absent_in_scope",
    }


def evidence_catalog(reader):
    entries = {}
    for case in cases(reader):
        for evidence in case.evidence:
            entry = entries.setdefault(
                evidence.name,
                {
                    "name": evidence.name,
                    "count": 0,
                    "types": set(),
                    "visibilities": set(),
                    "examples": [],
                },
            )
            entry["count"] += 1
            entry["types"].add(evidence.type)
            entry["visibilities"].add(evidence.input_visibility)
            if len(entry["examples"]) < 3:
                entry["examples"].append({"case_id": case.case_id, "evidence_id": evidence.id})
    return [
        {**row, "types": sorted(row["types"]), "visibilities": sorted(row["visibilities"])}
        for _, row in sorted(entries.items())
    ]


def field_values(reader, name):
    rows = []
    for case in cases(reader):
        rows.extend(
            {
                "case_id": case.case_id,
                "evidence_id": item.id,
                "value": item.value,
                "type": item.type,
                "input_visibility": item.input_visibility,
            }
            for item in case.evidence
            if item.name == name
        )
    return rows


def validate_evidence(reader, references, *, inputs=False):
    keys = [(ref.case_id, ref.evidence_id) for ref in references]
    if len(keys) != len(set(keys)):
        raise ValueError("Evidence references must be unique")
    for ref in references:
        evidence = reader.get_evidence(ref.case_id, ref.evidence_id)
        if inputs and evidence.input_visibility != "model_visible":
            raise ValueError("Task input evidence must be explicitly model_visible")


def build_scenarios(plan, *, max_scenarios=None):
    if set(plan.axis_assessments) != set(AXES):
        raise ValueError("Record all three Axis assessments before building combinations")
    values = [[row.value for row in plan.axis_assessments[axis] if row.supported] for axis in AXES]
    combinations = 1
    for axis_values in values:
        combinations *= len(axis_values)
    if max_scenarios is not None and combinations > max_scenarios:
        raise ValueError(
            f"More than {max_scenarios} combinations; narrow the supported Axis values first"
        )
    scenarios = {}
    for combination in itertools.product(*values):
        identifier = "scenario-" + hashlib.sha256(json.dumps(combination).encode()).hexdigest()[:16]
        scenarios[identifier] = plan.scenarios.get(identifier) or SemanticScenario(
            scenario_id=identifier, **dict(zip(AXES, combination, strict=True))
        )
    invalid = {task.task_id for task in plan.tasks.values() if task.scenario_id not in scenarios}
    if invalid - plan.exclusions.keys():
        raise ValueError("Exclude affected tasks before removing their Scenario combinations")
    # Keep historical excluded-task scenarios as explicit exclusions, not active combinations.
    for task_id in invalid:
        old = plan.scenarios[plan.tasks[task_id].scenario_id].model_copy(deep=True)
        old.decision = "exclude"
        old.reason = plan.exclusions[task_id].reason
        old.evidence_refs = plan.exclusions[task_id].evidence_refs
        scenarios[old.scenario_id] = old
    plan.scenarios = scenarios


def validate_plan(plan: TaskPlan, reader, *, publish=False):
    if plan.dataset.model_dump() != dataset_ref(reader):
        raise ValueError("Plan belongs to a different Dataset revision")
    for axis, assessments in plan.axis_assessments.items():
        values = [row.value for row in assessments]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate Axis values: {axis}")
        for row in assessments:
            validate_evidence(reader, row.evidence_refs)
    for scenario in plan.scenarios.values():
        validate_evidence(reader, scenario.evidence_refs)
        if scenario.decision != "pending" and (not scenario.reason or not scenario.evidence_refs):
            raise ValueError("Scenario decisions require a reason and evidence")
    for requirement in plan.requirements.values():
        if requirement.dataset != plan.dataset:
            raise ValueError("DataRequirement is bound to a different Dataset revision")
        for case_id in requirement.case_ids:
            reader.get_case(case_id)
        validate_evidence(reader, requirement.evidence_refs)
        if publish and set(requirement.affected_task_ids) - plan.tasks.keys():
            raise ValueError("DataRequirement references an unknown affected task")
    for acceptance in plan.acceptances.values():
        validate_evidence(reader, acceptance.evidence_refs)
    for exclusion in plan.exclusions.values():
        validate_evidence(reader, exclusion.evidence_refs)
    for task in plan.tasks.values():
        if task.dataset != plan.dataset or task.scenario_id not in plan.scenarios:
            raise ValueError("Task Dataset or Scenario does not match the plan")
        validate_evidence(reader, task.evidence_refs)
        validate_evidence(reader, task.input_evidence_refs, inputs=True)
        validate_evidence(reader, task.reference_evidence_refs)
        if set(task.requirement_ids) - plan.requirements.keys():
            raise ValueError("Task references an unknown DataRequirement")
        if task.task_id in plan.exclusions:
            continue
        if plan.scenarios[task.scenario_id].decision != "keep":
            raise ValueError("Active task must belong to a kept Scenario")
        if task.feasibility == "feasible":
            if (
                not task.evidence_refs
                or not task.input_evidence_refs
                or not task.reference_evidence_refs
            ):
                raise ValueError("Feasible task requires evidence for its inputs and reference")
            if set(task.requirement_ids) - plan.acceptances.keys():
                raise ValueError(
                    "Accept the outstanding data requirements before marking a task feasible"
                )
        if task.feasibility == "needs_data" and not task.requirement_ids:
            raise ValueError("needs_data task requires a DataRequirement")
    if publish:
        if set(plan.axis_assessments) != set(AXES):
            raise ValueError("Published plan must assess all three Axes")
        supported = [
            {row.value for row in plan.axis_assessments[axis] if row.supported} for axis in AXES
        ]
        count = 1
        for values in supported:
            count *= len(values)
        actual = {
            (s.input_structure, s.operation, s.clinical_target) for s in plan.scenarios.values()
        }
        current = {
            values
            for values in actual
            if all(value in allowed for value, allowed in zip(values, supported, strict=True))
        }
        if len(actual) != len(plan.scenarios):
            raise ValueError("Duplicate Scenario combinations")
        if len(current) != count:
            raise ValueError("Build and assess all supported Axis combinations before publication")
        if any(
            (s.input_structure, s.operation, s.clinical_target) not in current
            and s.decision != "exclude"
            for s in plan.scenarios.values()
        ):
            raise ValueError("Scenario is no longer supported by the current Axis assessments")
        if any(s.decision == "pending" for s in plan.scenarios.values()):
            raise ValueError("Complete Scenario decisions before publication")
        kept = {s.scenario_id for s in plan.scenarios.values() if s.decision == "keep"}
        if kept - {task.scenario_id for task in plan.tasks.values()}:
            raise ValueError("Each kept Scenario needs a task or an explicit exclusion decision")
        if any(
            task.feasibility == "proposed"
            for key, task in plan.tasks.items()
            if key not in plan.exclusions
        ):
            raise ValueError("Published tasks need an explicit feasibility assessment")
    return plan


def plan_summary(plan):
    return {
        "plan_id": plan.plan_id,
        "revision": plan.revision,
        "tasks": len(plan.tasks),
        "scenarios": len(plan.scenarios),
        "feasibility": dict(Counter(task.feasibility for task in plan.tasks.values())),
        "excluded_task_ids": list(plan.exclusions),
        "pending_requirement_ids": sorted(pending_requirements(plan)),
    }


def active_requirements(plan):
    return {
        key for key, requirement in plan.requirements.items() if not requirement.affected_task_ids
    } | {
        key
        for task_id, task in plan.tasks.items()
        if task_id not in plan.exclusions
        for key in task.requirement_ids
    }


def pending_requirements(plan):
    return active_requirements(plan) - plan.acceptances.keys()
