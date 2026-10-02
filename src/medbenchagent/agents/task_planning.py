"""Resumable, evidence-staged Task Planning Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.data.task_facts import build_dataset_facts, case_facts
from medbenchagent.planning.node_builder import build_task_scenarios
from medbenchagent.planning.access import DatasetAccessReceipt
from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
from medbenchagent.planning.task_discovery import leaf_identity, validate_candidate_leaf
from medbenchagent.planning.task_space import (
    AXIS_VALUES,
    AxisAnalysis,
    AxisValueDecision,
    CapabilityCategory,
    EvidenceBasis,
    EvidenceSelector,
    LeafTask,
    ScenarioReview,
    TaskAxis,
    TaskPlanState,
    TaskParadigm,
    OutputSpec,
)
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import NoArguments


class AxisArguments(NoArguments):
    axis: TaskAxis
    assessments: list[AxisValueDecision] = Field(min_length=1)


class AxisAnalysisArguments(NoArguments):
    analysis: dict[str, Any]


class ScenarioArguments(NoArguments):
    scenario_id: str | None = None


class LeafTaskDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leaf_task_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    question_template: str = Field(min_length=1)
    input: list[EvidenceSelector] = Field(min_length=1)
    target: str = Field(min_length=1)
    reference_rule: str = Field(min_length=1)
    resolver: ResolverSpec
    output_spec: OutputSpec
    task_paradigm: TaskParadigm
    capability_category: CapabilityCategory
    evidence: list[EvidenceBasis] = Field(default_factory=list)


class LeafTaskArguments(NoArguments):
    leaf: LeafTaskDraft


class ScenarioReviewArguments(NoArguments):
    review: ScenarioReview


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _access(context: dict[str, Any]) -> dict[str, Any]:
    return context.get("dataset_access_manifest", {})


def _profile(context: dict[str, Any]) -> dict[str, Any]:
    return _read(_access(context)["profile_path"])


def _plan_path(context: dict[str, Any]) -> Path:
    return Path(context.get("task_plan_path") or context.get("task_space_plan_path")
                or Path(context["operation_directory"]) / "task_plan.json")


def _evidence_dir(context: dict[str, Any]) -> Path:
    return Path(context["operation_directory"]) / "evidence"


def _load_plan(context: dict[str, Any]) -> TaskPlanState:
    path = _plan_path(context)
    if path.is_file():
        return TaskPlanState.model_validate(_read(path))
    return TaskPlanState(
        dataset={"dataset_id": context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))}
    )


def _write_json(path: Path, value: Any) -> str:
    atomic_write_json(path, value)
    return str(path)


def _write_evidence_artifacts(context: dict[str, Any], facts: dict[str, Any]) -> dict[str, str]:
    directory = _evidence_dir(context)
    directory.mkdir(parents=True, exist_ok=True)
    access = _access(context)
    receipt = DatasetAccessReceipt.model_validate({
        "schema_version": "dataset-access.v2",
        "dataset_id": context.get("dataset_scope", context.get("selected_dataset_id")),
        "revision": access.get("revision"),
        "cases_path": access.get("cases_path"),
        "asset_manifest_path": access.get("asset_manifest_path"),
        "profile_path": access.get("profile_path"),
        "duckdb_path": access.get("duckdb_path"),
        "annotation_digest": access.get("annotation_digest"),
        "asset_manifest_digest": access.get("asset_manifest_digest"),
        "profile_digest": access.get("profile_digest"),
        "duckdb_digest": access.get("duckdb_digest"),
        "case_count": facts.get("case_count", 0),
        "patient_count": facts.get("patient_count", 0),
        "patient_identity": facts.get("patient_identity", "unknown"),
    }).model_dump(mode="json")
    overview = {
        "schema_version": "dataset-overview.v2",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "case_count": facts.get("case_count", 0),
        "patient_count": facts.get("patient_count", 0),
        "patient_identity": facts.get("patient_identity", "unknown"),
        "assets": facts.get("assets", {}),
        "relations": facts.get("relations", {}),
        "evidence_counts": facts.get("evidence", {}),
    }
    catalog = {
        "schema_version": "evidence-catalog.v2",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "entries": list(facts.get("evidence_catalog", {}).values()),
        "names": facts.get("evidence_names", []),
        "annotation_mapping": facts.get("annotation_mapping", []),
    }
    relations = {
        "schema_version": "relation-summary.v2",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "relations": facts.get("relations", {}),
        "assets": facts.get("assets", {}),
    }
    samples_path = directory / "case_samples.jsonl"
    sample = facts.get("representative_case")
    if sample:
        samples_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        samples_path.write_text("", encoding="utf-8")
    paths = {
        "access": _write_json(directory.parent / "access.json", receipt),
        "overview": _write_json(directory / "overview.json", overview),
        "catalog": _write_json(directory / "catalog.json", catalog),
        "relations": _write_json(directory / "relations.json", relations),
        "case_samples": str(samples_path),
    }
    from medbenchagent.data.sources import digest_file
    paths["digests"] = {
        key: digest_file(value) for key, value in paths.items()
        if key != "case_samples" and Path(value).is_file()
    }
    manifest = {
        "schema_version": "planning-evidence.v2",
        "dataset_id": receipt["dataset_id"],
        "revision": receipt["revision"],
        "artifacts": paths,
    }
    paths["manifest"] = _write_json(directory / "manifest.json", manifest)
    return paths


def _write_artifacts(context: dict[str, Any], plan: TaskPlanState) -> dict[str, Any]:
    directory = Path(context["operation_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = _plan_path(context)
    _write_json(plan_path, plan.model_dump(mode="json"))
    axis_path = directory / "axis_decisions.json"
    scenarios_path = directory / "task_scenarios.json"
    leaves_path = directory / "leaf_tasks.json"
    _write_json(axis_path, {key: [item.model_dump(mode="json") for item in value] for key, value in plan.axis_decisions.items()})
    _write_json(scenarios_path, {key: value.model_dump(mode="json") for key, value in plan.scenarios.items()})
    _write_json(leaves_path, {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()})
    legacy_view = plan.model_dump(mode="json")
    legacy_view.update({
        "pending_node_ids": plan.pending_scenario_ids,
        "current_node_id": plan.current_scenario_id,
        "nodes": {key: value.model_dump(mode="json") for key, value in plan.scenarios.items()},
        "leaves": {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()},
    })
    return {
        "task_plan_path": str(plan_path),
        "task_space_plan_path": str(plan_path),
        "task_plan": plan.model_dump(mode="json"),
        "task_space_plan": legacy_view,
        "axis_decisions_path": str(axis_path),
        "task_scenarios_path": str(scenarios_path),
        "leaf_tasks_path": str(leaves_path),
        "task_nodes_path": str(scenarios_path),
        "task_scenarios": {key: value.model_dump(mode="json") for key, value in plan.scenarios.items()},
        "leaf_tasks": {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()},
        # Transitional in-process views; new persisted artifacts use the names above.
        "task_nodes": {key: value.model_dump(mode="json") for key, value in plan.scenarios.items()},
        "task_leaves": {key: value.model_dump(mode="json") for key, value in plan.leaf_tasks.items()},
    }


_write_derived_artifacts = _write_artifacts


def _initialize_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    context = dict(context)
    if _plan_path(context).is_file():
        plan = _load_plan(context)
        context.update(_write_artifacts(context, plan))
        return context
    access = _access(context)
    if access.get("schema_version") != "annotation.v5" or not access.get("cases_path"):
        plan = TaskPlanState(
            dataset={"dataset_id": context.get("dataset_scope", "dataset")},
            phase="blocked",
            review={"reason": "Dataset Access manifest is unavailable"},
        )
        context.update(_write_artifacts(context, plan))
        return context

    from medbenchagent.data.integrity import verify_prepared_indexes
    from medbenchagent.data.v5_index import V5DatasetQuery

    try:
        verify_prepared_indexes(access)
        query = V5DatasetQuery(access)
        cases = list(query.iter_cases())
        assets = {row["asset_id"]: row for row in query.assets()}
        facts = build_dataset_facts(cases, assets, profile=_profile(context), revision=access.get("revision"))
    except (OSError, ValueError, KeyError) as exc:
        plan = TaskPlanState(
            dataset={"dataset_id": context.get("dataset_scope", "dataset")},
            phase="blocked",
            review={"reason": f"Dataset Access verification failed: {exc}"},
        )
        context.update(_write_artifacts(context, plan))
        return context

    evidence_paths = _write_evidence_artifacts(context, facts)
    plan = TaskPlanState(
        dataset={"dataset_id": context.get("dataset_scope", "dataset")},
        access_path=evidence_paths["access"],
        evidence_manifest_path=evidence_paths["manifest"],
        phase="axes" if facts.get("case_count", 0) and facts.get("evidence_names") else "blocked",
        review={} if facts.get("case_count", 0) and facts.get("evidence_names") else {"reason": "Dataset has no Case or Evidence"},
    )
    context.update(_write_artifacts(context, plan))
    context["dataset_facts"] = facts
    context["planning_context"] = {
        "evidence_manifest_path": evidence_paths["manifest"],
        "overview_path": evidence_paths["overview"],
        "catalog_path": evidence_paths["catalog"],
        "relations_path": evidence_paths["relations"],
    }
    return context


def _axis_order() -> tuple[TaskAxis, ...]:
    return (TaskAxis.INPUT_STRUCTURE, TaskAxis.OPERATION, TaskAxis.CLINICAL_TARGET)


def _facts(context: dict[str, Any], plan: TaskPlanState) -> dict[str, Any]:
    facts = context.get("dataset_facts")
    if facts:
        return facts
    legacy_facts = getattr(plan, "dataset_facts", None)
    if legacy_facts:
        return legacy_facts
    manifest = plan.evidence_manifest_path
    if manifest:
        paths = _read(manifest).get("artifacts", {})
        return _read(paths["overview"]) | {"evidence_catalog": _read(paths["catalog"]).get("entries", [])}
    try:
        profile = _profile(context)
    except (KeyError, OSError, ValueError):
        return {}
    evidence_catalog = {row["name"]: row for row in profile.get("evidence", []) if row.get("name")}
    evidence = {}
    for row in evidence_catalog.values():
        for kind, count in row.get("types", {}).items():
            evidence[kind] = evidence.get(kind, 0) + count
        evidence["numeric"] = evidence.get("numeric", 0) + row.get("value_types", {}).get("number", 0)
    assets = profile.get("assets", {})
    return {
        "schema_version": "dataset-facts.v2",
        "revision": _access(context).get("revision"),
        "case_count": profile.get("case_count", 0),
        "patient_count": profile.get("patient_count", 0),
        "patient_identity": "present" if profile.get("patient_count", 0) else "absent",
        "evidence_catalog": evidence_catalog,
        "evidence_names": list(evidence_catalog),
        "evidence": evidence,
        "assets": {
            "image_assets": assets.get("kinds", {}).get("image", 0),
            "primary_image_assets": assets.get("kinds", {}).get("image", 0),
            "volume_assets": assets.get("kinds", {}).get("volume", 0),
            "multi_image_cases": 0,
            "related_image_cases": 0,
        },
        "relations": {
            "paired_text": 0,
            "paired_study": 0,
            "comparable": 0,
            "related_image_groups": 0,
        },
    }


def _build_scenarios(context: dict[str, Any], plan: TaskPlanState) -> TaskPlanState:
    facts = _facts(context, plan)
    node_context = {
        "evidence": facts.get("evidence", {}),
        "assets": facts.get("assets", {}),
        "relations": facts.get("relations", {}),
        "evidence_names": facts.get("evidence_names", []),
        "annotation_mapping": facts.get("annotation_mapping", []),
    }
    access = _access(context)
    if access.get("schema_version") == "annotation.v5" and access.get("cases_path"):
        from medbenchagent.data.v5_index import V5DatasetQuery
        query = V5DatasetQuery(access)
        assets = {row["asset_id"]: row for row in query.assets()}
        node_context["case_facts"] = [case_facts(case, assets) for case in query.iter_cases()]
    analysis = AxisAnalysis(
        input_structure=plan.axis_decisions[TaskAxis.INPUT_STRUCTURE.value],
        operation=plan.axis_decisions[TaskAxis.OPERATION.value],
        clinical_target=plan.axis_decisions[TaskAxis.CLINICAL_TARGET.value],
        dataset_revision=str(facts.get("revision") or access.get("revision") or "1"),
    )
    scenarios = build_task_scenarios(analysis, node_context)
    active = sorted(key for key, value in scenarios.items() if not value.exclusion_reason)
    if not active:
        raise ValueError("Axis analysis produced no executable Task Scenario")
    plan.scenarios = scenarios
    plan.phase = "leaves"
    plan.pending_scenario_ids = active
    plan.completed_scenario_ids = []
    plan.current_scenario_id = None
    return plan


def analyze_axis(context: dict[str, Any], *, axis: TaskAxis, assessments: list[AxisValueDecision]) -> dict[str, Any]:
    axis = TaskAxis(axis)
    plan = _load_plan(context)
    if plan.phase != "axes":
        raise ValueError(f"Axis analysis is not available in phase {plan.phase}")
    expected = _axis_order()[len(plan.axis_decisions)]
    if axis != expected:
        raise ValueError(f"Expected Axis {expected}, received {axis}")
    assessments = [AxisValueDecision.model_validate(value) for value in assessments]
    allowed = set(AXIS_VALUES[axis])
    actual = [value.value for value in assessments]
    if set(actual) != allowed or len(actual) != len(set(actual)):
        raise ValueError(f"Axis analysis must cover registry exactly for {axis}")
    plan.axis_decisions[axis.value] = assessments
    plan.revision += 1
    if len(plan.axis_decisions) == len(_axis_order()):
        plan.next_axis = None
        plan = _build_scenarios(context, plan)
        summary = (
            f"Axis {axis.value} saved; built {len(plan.scenarios)} Task Scenarios "
            f"({len(plan.pending_scenario_ids)} executable)"
        )
    else:
        plan.next_axis = _axis_order()[len(plan.axis_decisions)]
        summary = (
            f"Axis {axis.value} saved; supported values: "
            + ", ".join(value.value for value in assessments if value.supported)
        )
    updates = _write_artifacts(context, plan)
    return result("success", summary, milestone={"kind": "axis", "axis": axis.value}, **updates)


def analyze_axes(context: dict[str, Any], *, analysis: dict[str, Any]) -> dict[str, Any]:
    """Import a complete AxisAnalysis for deterministic/in-process callers."""
    parsed = AxisAnalysis.model_validate(analysis)
    plan = _load_plan(context)
    plan.axis_decisions = {
        TaskAxis.INPUT_STRUCTURE.value: parsed.input_structure,
        TaskAxis.OPERATION.value: parsed.operation,
        TaskAxis.CLINICAL_TARGET.value: parsed.clinical_target,
    }
    plan.next_axis = None
    plan.phase = "axes"
    plan.revision += 1
    plan = _build_scenarios(context, plan)
    return result("success", "Axis decisions saved", **_write_artifacts(context, plan))


def build_nodes(context: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper; new plans build Task Scenarios after Axis decisions."""
    plan = _load_plan(context)
    if not plan.axis_decisions:
        path = context.get("axis_analysis_path")
        if not path:
            raise ValueError("Axis decisions are required before building Scenarios")
        parsed = AxisAnalysis.model_validate(_read(path))
        plan.axis_decisions = {
            TaskAxis.INPUT_STRUCTURE.value: parsed.input_structure,
            TaskAxis.OPERATION.value: parsed.operation,
            TaskAxis.CLINICAL_TARGET.value: parsed.clinical_target,
        }
    plan.phase = "axes"
    plan = _build_scenarios(context, plan)
    return result("success", "Task Scenarios saved", **_write_artifacts(context, plan))


def read_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    return result("success", json.dumps({
        "schema_version": "planning-index.v2",
        "phase": plan.phase,
        "next_axis": plan.next_axis,
        "evidence_manifest_path": plan.evidence_manifest_path,
        "axis_decisions": list(plan.axis_decisions),
        "scenario_count": len(plan.scenarios),
        "leaf_task_count": len(plan.leaf_tasks),
    }, ensure_ascii=False))


def read_dataset_overview(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    if not plan.evidence_manifest_path:
        raise ValueError("Planning evidence manifest is unavailable")
    paths = _read(plan.evidence_manifest_path)["artifacts"]
    overview = _read(paths["overview"])
    return result(
        "success",
        f"Access verified for {overview.get('dataset_id')} revision {overview.get('revision')} with {overview.get('case_count', 0)} Cases",
        milestone={"kind": "access", "dataset_id": overview.get("dataset_id"), "revision": overview.get("revision")},
        overview=json.dumps(overview, ensure_ascii=False),
    )


class EvidenceCatalogArguments(NoArguments):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)
    names: list[str] = Field(default_factory=list)


class AxisEvidenceArguments(NoArguments):
    axis: TaskAxis


def read_evidence_catalog(context: dict[str, Any], *, offset=0, limit=20, names=None) -> dict[str, Any]:
    plan = _load_plan(context)
    paths = _read(plan.evidence_manifest_path)["artifacts"]
    entries = _read(paths["catalog"]).get("entries", [])
    if names:
        entries = [row for row in entries if row.get("name") in set(names)]
    page = entries[offset:offset + limit]
    return result("success", json.dumps({
        "revision": _read(paths["catalog"]).get("revision"),
        "entries": page,
        "total": len(entries),
        "next_offset": offset + limit if offset + limit < len(entries) else None,
    }, ensure_ascii=False))


def read_axis_evidence(context: dict[str, Any], *, axis: TaskAxis) -> dict[str, Any]:
    plan = _load_plan(context)
    facts = _facts(context, plan)
    evidence = {
        "schema_version": "axis-evidence.v2",
        "axis": axis.value,
        "revision": facts.get("revision"),
        "allowed_values": list(AXIS_VALUES[axis]),
        "dataset_overview": {
            "case_count": facts.get("case_count", 0),
            "patient_count": facts.get("patient_count", 0),
            "patient_identity": facts.get("patient_identity", "unknown"),
            "assets": facts.get("assets", {}),
            "relations": facts.get("relations", {}),
            "evidence": facts.get("evidence", {}),
        },
        "evidence_catalog": list(facts.get("evidence_catalog", {}).values())
        if isinstance(facts.get("evidence_catalog"), dict) else facts.get("evidence_catalog", []),
    }
    axis_dir = Path(context["operation_directory"]) / "axis_evidence"
    path = axis_dir / f"{axis.value}.json"
    _write_json(path, evidence)
    return result("success", json.dumps(evidence, ensure_ascii=False), evidence_path=str(path))


def read_scenario_context(context: dict[str, Any], *, scenario_id: str | None = None) -> dict[str, Any]:
    plan = _load_plan(context)
    scenario_id = scenario_id or plan.current_scenario_id or (plan.pending_scenario_ids[0] if plan.pending_scenario_ids else None)
    if not scenario_id or scenario_id not in plan.scenarios:
        raise ValueError("No pending Task Scenario is available")
    if plan.current_scenario_id != scenario_id:
        scenario = plan.scenarios[scenario_id]
        plan.current_scenario_id = scenario_id
        plan.scenarios[scenario_id] = scenario.model_copy(update={"status": "active"})
        _write_artifacts(context, plan)
    from medbenchagent.data.v5_index import V5DatasetQuery
    context_value = V5DatasetQuery(_access(context)).node_context(plan.scenarios[scenario_id])
    context_value["leaf_task_ids"] = plan.scenarios[scenario_id].leaf_task_ids
    context_value["plan_phase"] = plan.phase
    return result("success", json.dumps(context_value, ensure_ascii=False))


def save_leaf_task(context: dict[str, Any], *, leaf: LeafTaskDraft) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves" or not plan.current_scenario_id:
        raise ValueError("An active Task Scenario is required before saving a Leaf Task")
    parsed = LeafTask.model_validate(leaf.model_dump(mode="json"))
    if parsed.scenario_id != plan.current_scenario_id:
        raise ValueError("Leaf Task must belong to the active Scenario")
    scenario = plan.scenarios[parsed.scenario_id]
    if parsed.capability_category != scenario.capability_category:
        raise ValueError("Leaf Task Category must match its Scenario")
    evidence_names = set(_facts(context, plan).get("evidence_names", []))
    evidence_names.update(
        selector.name for selector in parsed.input
        if selector.name and selector.evidence_type == "asset"
    )
    validate_candidate_leaf(parsed, plan.scenarios, resolver_context={"evidence_names": evidence_names})
    identity = leaf_identity(parsed)
    if any(key != parsed.leaf_id and leaf_identity(value) == identity for key, value in plan.leaf_tasks.items()):
        raise ValueError("Duplicate Leaf Task semantics")
    from medbenchagent.data.query import DatasetQuery
    query = DatasetQuery(_access(context))
    if hasattr(query, "iter_cases"):
        cases = list(query.iter_cases())
    else:
        page = query.sample(offset=0, limit=512)
        cases = list(page.get("values", []))
    eligible = sum(
        not isinstance(resolve(case, ResolverSpec.parse(parsed.resolver)), NotApplicable)
        for case in cases
    )
    if eligible <= 0:
        raise ValueError("Leaf Task resolver does not resolve any Case")
    parsed = parsed.model_copy(update={"eligible_count": eligible})
    plan.leaf_tasks[parsed.leaf_id] = parsed
    plan.revision += 1
    scenario = plan.scenarios[parsed.scenario_id]
    plan.scenarios[parsed.scenario_id] = scenario.model_copy(
        update={
            "leaf_task_ids": [*scenario.leaf_task_ids, parsed.leaf_id],
            "status": "review_pending",
        }
    )
    updates = _write_artifacts(context, plan)
    return result(
        "success",
        f"Leaf Task {parsed.leaf_id} saved ({eligible} eligible Cases)",
        milestone={"kind": "leaf", "leaf_task_id": parsed.leaf_id, "capacity": eligible},
        **updates,
    )


def save_task_leaf(context: dict[str, Any], *, leaf: dict[str, Any]) -> dict[str, Any]:
    """Compatibility entry point for callers using the old tool name."""
    plan = _load_plan(context)
    if plan.phase != "leaves":
        plan.phase = "leaves"
    if not plan.current_scenario_id:
        current = getattr(plan, "current_node_id", None) or (plan.pending_scenario_ids[0] if plan.pending_scenario_ids else None)
        if current:
            plan.current_scenario_id = current
            _write_artifacts(context, plan)
    parsed = LeafTask.model_validate(leaf)
    normalized = {
        "leaf_task_id": parsed.leaf_id,
        "name": parsed.name,
        "scenario_id": parsed.scenario_id,
        "question_template": parsed.question_template,
        "input": [selector.model_dump(mode="json") for selector in parsed.input],
        "target": parsed.target,
        "reference_rule": parsed.reference_rule,
        "resolver": parsed.resolver,
        "output_spec": parsed.output_spec.value,
        "task_paradigm": parsed.task_paradigm.value,
        "capability_category": parsed.capability_category.value,
        "evidence": [basis.model_dump(mode="json") for basis in parsed.evidence],
    }
    return save_leaf_task(context, leaf=LeafTaskDraft.model_validate(normalized))


def review_current_scenario(context: dict[str, Any], *, review: ScenarioReview) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves" or not plan.current_scenario_id:
        raise ValueError("An active Scenario is required before review")
    if review.scenario_id != plan.current_scenario_id:
        raise ValueError("Review must belong to the active Scenario")
    scenario = plan.scenarios[review.scenario_id]
    expected = set(scenario.leaf_task_ids)
    if set(review.checked_leaf_task_ids) != expected or not expected:
        raise ValueError("Scenario review must cover every saved Leaf Task")
    if review.status == "approved":
        if any((plan.leaf_tasks[key].eligible_count or 0) <= 0 for key in expected):
            raise ValueError("Cannot approve a Scenario with a zero-capacity Leaf Task")
        plan.scenarios[review.scenario_id] = scenario.model_copy(update={"status": "completed"})
        plan.reviews[review.scenario_id] = review
        plan.completed_scenario_ids = sorted(set(plan.completed_scenario_ids) | {review.scenario_id})
        plan.pending_scenario_ids = [key for key in plan.pending_scenario_ids if key != review.scenario_id]
        plan.current_scenario_id = None
        if not plan.pending_scenario_ids:
            plan.phase = "complete"
    else:
        plan.scenarios[review.scenario_id] = scenario.model_copy(update={"status": "active"})
        plan.reviews[review.scenario_id] = review
    plan.revision += 1
    updates = _write_artifacts(context, plan)
    return result(
        "success",
        f"Scenario {review.scenario_id} review saved",
        milestone={"kind": "scenario_review", "scenario_id": review.scenario_id, "status": review.status},
        **updates,
    )


def review_current_node(context: dict[str, Any], *, review: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the pre-v3 Scenario review tool."""
    parsed = dict(review)
    scenario_id = parsed.pop("scenario_id", parsed.pop("node_id", None))
    checked = parsed.pop("checked_leaf_task_ids", parsed.pop("checked_leaf_ids", []))
    return review_current_scenario(
        context,
        review=ScenarioReview(
            scenario_id=scenario_id,
            checked_leaf_task_ids=checked,
            status=parsed.get("status", "rejected"),
            issues=parsed.get("issues", []),
        ),
    )


def classify_task_nodes(context: dict[str, Any], *, assignments: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility no-op: Capability Category is derived from Operation."""
    plan = _load_plan(context)
    if plan.phase == "leaves" and not plan.pending_scenario_ids and not plan.current_scenario_id:
        plan.phase = "complete"
    return result("success", "Capability Categories are derived from Operation", **_write_artifacts(context, plan))


def finish_task_plan(context: dict[str, Any]) -> dict[str, Any]:
    require(context, "Dataset", "PlanningConfig")
    plan = _load_plan(context)
    if plan.phase != "complete" or plan.pending_scenario_ids or plan.current_scenario_id:
        raise ValueError("Task Plan still has pending Scenarios")
    for scenario_id, scenario in plan.scenarios.items():
        if scenario.exclusion_reason:
            continue
        if not scenario.leaf_task_ids:
            raise ValueError(f"Scenario has no Leaf Task: {scenario_id}")
        review = plan.reviews.get(scenario_id)
        if review is None or review.status != "approved":
            raise ValueError(f"Scenario lacks an approved review: {scenario_id}")
    updates = _write_artifacts(context, plan)
    versions = publish(
        context, "Plan", {"task_plan": updates["task_plan_path"]},
        ("Dataset", "PlanningConfig"),
    ) if context.get("run_directory") else {}
    return result(
        "complete",
        "Task Plan published",
        working_blueprint_path=updates["task_plan_path"],
        milestone={"kind": "plan_complete"},
        **updates,
        **versions,
    )


def _available_tools(context: dict[str, Any]) -> set[str]:
    plan = _load_plan(context)
    if plan.phase in {"access", "overview"}:
        return {"read_planning_context", "read_dataset_overview", "read_evidence_catalog"}
    if plan.phase == "axes":
        return {
            "read_planning_context", "read_dataset_overview", "read_evidence_catalog",
            "read_axis_evidence", "analyze_axis",
        }
    if plan.phase == "leaves":
        return {"read_scenario_context", "save_leaf_task", "review_current_scenario"}
    if plan.phase == "complete":
        return {"finish"}
    if plan.phase == "blocked":
        return {"read_planning_context", "read_dataset_overview"}
    return set()


finish_task_space = finish_task_plan


def compile_task_planning_agent(*, checkpointer=None):
    tools = (
        read_planning_context,
        read_dataset_overview,
        read_evidence_catalog,
        read_axis_evidence,
        analyze_axis,
        read_scenario_context,
        save_leaf_task,
        review_current_scenario,
    )
    return compile_agent(
        "TaskPlanningAgent",
        "Build one resumable annotation.v5 Task Plan using six levels: Task Axis, Axis Value, Task Scenario, Leaf Task, Task Paradigm and Benchmark Item. Axis definitions are fixed; assess every registered value exactly once, in order input_structure, operation, clinical_target. Code constructs Scenarios and assigns Capability Category from Operation using the registered six-category mapping. In the Leaves phase inspect one current Scenario, propose concrete evidence-backed Leaf Tasks with a valid Resolver and Task Paradigm, then review that Scenario. Never invent Axis Values, use reference evidence as model input, or expose chain-of-thought. Read only the evidence slice required by the current phase.",
        {tool.__name__: tool for tool in tools},
        lambda c: {
            "task_plan": str(_plan_path(c)),
            "phase": _load_plan(c).phase,
            "next_axis": _load_plan(c).next_axis,
            "current_scenario_id": _load_plan(c).current_scenario_id,
            "pending_scenario_count": len(_load_plan(c).pending_scenario_ids),
            "evidence_manifest_path": _load_plan(c).evidence_manifest_path,
        },
        finish_task_plan,
        args_schemas={
            "read_evidence_catalog": EvidenceCatalogArguments,
            "read_axis_evidence": AxisEvidenceArguments,
            "analyze_axis": AxisArguments,
            "read_scenario_context": ScenarioArguments,
            "save_leaf_task": LeafTaskArguments,
            "review_current_scenario": ScenarioReviewArguments,
        },
        available_tools=_available_tools,
        initialize_context=_initialize_planning_context,
        checkpointer=checkpointer,
    )


def apply_template_feedback(context, feedback):
    return result("blocked", "Template review is part of Task Planning")
