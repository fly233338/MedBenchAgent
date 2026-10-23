"""Deterministic semantic planning operations over the canonical Case contract."""

import hashlib
import itertools
import json
from collections import Counter

from .contracts import SemanticScenario, TaskPlan, ProfileEvidenceRef


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
    keys = [ref.model_dump_json() for ref in references]
    if len(keys) != len(set(keys)):
        raise ValueError("Evidence references must be unique")
    for ref in references:
        if isinstance(ref, ProfileEvidenceRef):
            if not hasattr(reader, "get_observation"):
                raise ValueError("This planning input has no Profile observations")
            reader.get_observation(ref.observation_id)
            continue
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


def validate_plan(plan: TaskPlan, reader, *, publish=False, task_data=None):
    from medbenchagent.generation.backends import BACKENDS, validate_leaf
    if plan.dataset.model_dump() != dataset_ref(reader):
        raise ValueError("Plan belongs to a different Dataset revision")
    for axis, assessments in plan.axis_assessments.items():
        values = [row.value for row in assessments]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate Axis values: {axis}")
        for row in assessments:
            validate_evidence(reader, row.evidence_refs)
    for scenario in plan.scenarios.values():
        if set(scenario.backend_ids) - BACKENDS.keys():
            raise ValueError("Scenario selects an unavailable backend")
        if set(scenario.requirement_ids) - plan.requirements.keys():
            raise ValueError("Scenario references an unknown DataRequirement")
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
        if set(requirement.affected_scenario_ids) - plan.scenarios.keys():
            raise ValueError("DataRequirement references an unknown Scenario")
        if requirement.backend_id is not None and requirement.backend_id not in BACKENDS:
            raise ValueError("DataRequirement selects an unavailable backend")
    for acceptance in [*plan.acceptances.values(), *plan.task_data_acceptances.values()]:
        accepted_reader = (task_data or {}).get(acceptance.artifact_id)
        if accepted_reader is None or accepted_reader.content_digest != acceptance.digest:
            raise ValueError("Bind accepted TaskData to validate its evidence")
        for evidence in acceptance.evidence_refs:
            validate_evidence(reader if isinstance(evidence, ProfileEvidenceRef) else accepted_reader, [evidence])
    for exclusion in plan.exclusions.values():
        validate_evidence(reader, exclusion.evidence_refs)
    for task in plan.tasks.values():
        if task.dataset != plan.dataset or task.scenario_id not in plan.scenarios:
            raise ValueError("Task Dataset or Scenario does not match the plan")
        validate_evidence(reader, task.evidence_refs)
        task_reader = reader
        if task.task_data is not None:
            task_reader = (task_data or {}).get(task.task_data.artifact_id)
            if task_reader is None or task_reader.content_digest != task.task_data.digest:
                raise ValueError("Bind the Leaf's exact TaskData input")
            if task_reader.manifest.dataset != plan.dataset:
                raise ValueError("TaskData belongs to another source Dataset")
        validate_evidence(task_reader, task.input_evidence_refs, inputs=True)
        validate_evidence(task_reader, task.reference_evidence_refs)
        if set(task.requirement_ids) - plan.requirements.keys():
            raise ValueError("Task references an unknown DataRequirement")
        if task.task_id in plan.exclusions:
            continue
        if plan.scenarios[task.scenario_id].decision != "keep":
            raise ValueError("Active task must belong to a kept Scenario")
        if task.feasibility == "feasible":
            if not task.evidence_refs or (task.assembly is None and (not task.input_evidence_refs or not task.reference_evidence_refs)):
                raise ValueError("Feasible task requires evidence for its inputs and reference")
            if task.assembly is not None:
                accepted = plan.task_data_acceptances.get(task.task_data.artifact_id) if task.task_data else None
                if accepted is None or accepted.digest != task.task_data.digest:
                    raise ValueError("Accept TaskData before publishing an executable Leaf")
                if task.assembly.backend_id not in plan.scenarios[task.scenario_id].backend_ids:
                    raise ValueError("Leaf backend is not selected by its Scenario")
                validate_leaf(task, task_reader)
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
        key for key, requirement in plan.requirements.items()
        if (not requirement.affected_task_ids and not requirement.affected_scenario_ids)
        or any(plan.scenarios[scenario].decision == "keep" for scenario in requirement.affected_scenario_ids)
    } | {
        key
        for task_id, task in plan.tasks.items()
        if task_id not in plan.exclusions
        for key in task.requirement_ids
    }


def pending_requirements(plan):
    return active_requirements(plan) - plan.acceptances.keys()


def scoped_objects(plan, scope):
    """Resolve explicit objects without expanding a single-task repair to its siblings."""
    scenarios, tasks, requirements = set(), set(), set()
    for item in scope:
        if item.kind == "artifact":
            return set(plan.scenarios), set(plan.tasks), set(plan.requirements)
        if item.kind == "scenario":
            scenarios.update(item.object_ids)
        elif item.kind == "task":
            tasks.update(item.object_ids)
        elif item.kind == "requirement":
            requirements.update(item.object_ids)
    scenarios.update(s for key in requirements for s in plan.requirements[key].affected_scenario_ids)
    tasks.update(key for key, task in plan.tasks.items() if task.scenario_id in scenarios)
    requirements.update(key for key, r in plan.requirements.items()
                        if set(r.affected_scenario_ids) & scenarios or set(r.affected_task_ids) & tasks)
    requirements.update(key for identifier in tasks for key in plan.tasks[identifier].requirement_ids)
    scenarios.update(plan.tasks[key].scenario_id for key in tasks)
    return scenarios, tasks, requirements


def check_operation_complete(plan, operation, scope, *, task_data=None):
    scenarios, tasks, requirements = scoped_objects(plan, scope)
    if operation == "discover_scenarios":
        if set(plan.axis_assessments) != set(AXES):
            raise ValueError("Assess all three Axes before completing discovery")
        for key in scenarios:
            scenario = plan.scenarios[key]
            if scenario.decision == "pending":
                raise ValueError(f"Scenario {key} still needs a decision")
            if scenario.decision != "keep":
                continue
            if not scenario.backend_ids:
                raise ValueError(f"Scenario {key} needs a backend; retain technical blockers")
            for backend in scenario.backend_ids:
                if not any(key in r.affected_scenario_ids and r.backend_id == backend
                           and r.required_fields for r in plan.requirements.values()) and not any(
                               identifier in plan.task_data_acceptances and reader.manifest.backend_id == backend
                               and key in reader.manifest.scenario_ids
                               for identifier, reader in (task_data or {}).items()):
                    raise ValueError(f"Scenario {key}/{backend} needs a concrete preparation requirement")
    elif operation == "design_leaves":
        if not tasks - plan.exclusions.keys():
            raise ValueError("No executable Leaf in the selected scope")
        for key in scenarios:
            if plan.scenarios[key].decision == "keep" and not any(
                task.scenario_id == key and (task.assembly or task.task_id in plan.exclusions)
                for task in plan.tasks.values()
            ):
                raise ValueError(f"Scenario {key} has no Leaf decision")
        for key in tasks - plan.exclusions.keys():
            task = plan.tasks[key]
            if task.feasibility != "feasible" or task.assembly is None:
                raise ValueError(f"Leaf {key} is not executable")
        if requirements & pending_requirements(plan):
            raise ValueError("Accept in-scope requirements before completing Leaf design")
    else:
        raise ValueError(f"Unsupported Planner operation: {operation}")
