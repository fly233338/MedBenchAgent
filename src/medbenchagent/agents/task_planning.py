"""Task Planning Agent for annotation.v5 datasets.

The agent owns one resumable Task Plan draft.  The model supplies semantic
judgements; deterministic code owns ordering, contracts, capacity and final
completeness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from medbenchagent.data.task_facts import build_dataset_facts, case_facts
from medbenchagent.planning.blueprint import blueprint_dimension_ids, load_blueprint
from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.resolver import NotApplicable, ResolverSpec, resolve
from medbenchagent.planning.task_discovery import leaf_identity, validate_candidate_leaf, validate_task_tree
from medbenchagent.planning.task_space import (
    AXIS_VALUES,
    AxisAnalysis,
    AxisAssessment,
    NodeReview,
    TaskAxis,
    TaskLeaf,
    TaskNode,
    TaskSpacePlan,
)
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import NoArguments


class AxisArguments(NoArguments):
    axis: str
    assessments: list[dict[str, Any]] = Field(min_length=1)


class AxisAnalysisArguments(NoArguments):
    """Compatibility input for callers that still provide a complete analysis."""

    analysis: dict[str, Any]


class TaskNodeArguments(NoArguments):
    node_id: str | None = None


class TaskLeafArguments(NoArguments):
    leaf: dict[str, Any]


class NodeReviewArguments(NoArguments):
    review: dict[str, Any]


class CapabilityAssignmentArguments(NoArguments):
    assignments: list[dict[str, Any]] = Field(min_length=1)


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _profile(context: dict[str, Any]) -> dict[str, Any]:
    return _read(context["dataset_access_manifest"]["profile_path"])


def _plan_path(context: dict[str, Any]) -> Path:
    return Path(context.get("task_space_plan_path") or Path(context["operation_directory"]) / "task_space_plan.json")


def _blueprint_path(context: dict[str, Any]) -> Path | None:
    explicit = context.get("blueprint_path")
    if explicit:
        return Path(explicit)
    try:
        from medbenchagent.settings import get_settings
        path = get_settings().paths.blueprint_template
        return Path(path)
    except (AttributeError, RuntimeError, ValueError):
        return None


def _load_blueprint_for_context(context: dict[str, Any]) -> dict[str, Any]:
    path = _blueprint_path(context)
    return load_blueprint(path) if path else {"name": "Blueprint", "revision": "local", "capability_dimensions": [], "task_concepts": []}


def _load_plan(context: dict[str, Any]) -> TaskSpacePlan:
    path = _plan_path(context)
    if path.is_file():
        return TaskSpacePlan.model_validate(_read(path))
    return TaskSpacePlan(
        dataset={"dataset_id": context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))},
        dataset_facts=context.get("dataset_facts", {}),
        blueprint=context.get("blueprint", {}),
    )


def _write_derived_artifacts(context: dict[str, Any], plan: TaskSpacePlan) -> dict[str, Any]:
    directory = Path(context["operation_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = _plan_path(context)
    atomic_write_json(plan_path, plan.model_dump(mode="json"))
    axis_path = directory / "axis_analysis.json"
    nodes_path = directory / "task_nodes.json"
    leaves_path = directory / "task_leaves.json"
    if plan.axis_analysis is not None:
        atomic_write_json(axis_path, plan.axis_analysis.model_dump(mode="json"))
    atomic_write_json(nodes_path, {key: value.model_dump(mode="json") for key, value in plan.nodes.items()})
    atomic_write_json(leaves_path, {key: value.model_dump(mode="json") for key, value in plan.leaves.items()})
    return {
        "task_space_plan_path": str(plan_path),
        "task_space_plan": plan.model_dump(mode="json"),
        "axis_analysis_path": str(axis_path) if plan.axis_analysis is not None else context.get("axis_analysis_path"),
        "task_nodes_path": str(nodes_path),
        "task_leaves_path": str(leaves_path),
        "task_nodes": {key: value.model_dump(mode="json") for key, value in plan.nodes.items()},
        "task_leaves": {key: value.model_dump(mode="json") for key, value in plan.leaves.items()},
        "dataset_facts": plan.dataset_facts,
        "blueprint": plan.blueprint,
    }


def _fact_context(context: dict[str, Any]) -> dict[str, Any]:
    facts = dict(context.get("dataset_facts", {}))
    if facts.get("evidence") and facts.get("assets"):
        return facts
    profile = _profile(context)
    evidence: dict[str, int] = {}
    for row in profile.get("evidence", []):
        for kind, count in row.get("types", {}).items():
            evidence[kind] = evidence.get(kind, 0) + count
        evidence["numeric"] = evidence.get("numeric", 0) + row.get("value_types", {}).get("number", 0)
    assets = profile.get("assets", {})
    per_case = {int(key): value for key, value in assets.get("assets_per_case", {}).items()}
    related = sum(value for size, value in per_case.items() if size >= 2)
    return {
        "schema_version": "dataset-facts",
        "revision": context.get("dataset_access_manifest", {}).get("revision"),
        "case_count": profile.get("case_count", 0),
        "patient_count": profile.get("patient_count", 0),
        "evidence_catalog": {row["name"]: row for row in profile.get("evidence", []) if row.get("name")},
        "evidence_names": [row["name"] for row in profile.get("evidence", []) if row.get("name")],
        "annotation_mapping": [
            {"role": f"evidence_{index}", "name": row["name"], "evidence_type": evidence_type}
            for index, row in enumerate(profile.get("evidence", []))
            for evidence_type in sorted(row.get("types", {}))
            if row.get("name") and evidence_type in {"value", "text", "asset", "region"}
        ],
        "evidence": evidence,
        "assets": {
            "volume_assets": assets.get("kinds", {}).get("volume", 0),
            "image_assets": assets.get("kinds", {}).get("image", 0),
            "related_image_cases": related,
            "multi_image_cases": related,
        },
        "relations": {
            "related": sum(row.get("has_ref", 0) for row in profile.get("evidence", [])),
            "paired": related,
            "paired_study": related,
            "paired_text": sum(row.get("types", {}).get("text", 0) for row in profile.get("evidence", [])),
        },
    }


def _initialize_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    """Create the one canonical draft before the model's first decision."""
    context = dict(context)
    if context.get("task_space_plan_path") and Path(context["task_space_plan_path"]).is_file():
        plan = _load_plan(context)
        context.update(_write_derived_artifacts(context, plan))
        return context
    access = context.get("dataset_access_manifest", {})
    if access.get("schema_version") != "annotation.v5" or not access.get("cases_path"):
        return context
    from medbenchagent.data.v5_index import V5DatasetQuery

    query = V5DatasetQuery(access)
    cases = list(query.iter_cases())
    assets = {row["asset_id"]: row for row in query.assets()}
    facts = build_dataset_facts(cases, assets, profile=_profile(context), revision=access.get("revision"))
    blueprint = _load_blueprint_for_context(context)
    plan = TaskSpacePlan(
        dataset={"dataset_id": context.get("dataset_scope", context.get("selected_dataset_id", "dataset"))},
        dataset_facts=facts,
        blueprint=blueprint,
        phase="axes" if facts.get("case_count", 0) and facts.get("evidence_names") else "blocked",
        review={} if facts.get("case_count", 0) and facts.get("evidence_names") else {"reason": "Dataset has no Case or Evidence"},
    )
    context.update(_write_derived_artifacts(context, plan))
    context["planning_context"] = facts
    return context


def _axis_order() -> tuple[TaskAxis, ...]:
    return (TaskAxis.INPUT_STRUCTURE, TaskAxis.OPERATION, TaskAxis.CLINICAL_TARGET)


def _advance_nodes(context: dict[str, Any], plan: TaskSpacePlan) -> TaskSpacePlan:
    facts = dict(plan.dataset_facts)
    node_context = {
        "evidence": facts.get("evidence", {}),
        "assets": facts.get("assets", {}),
        "relations": facts.get("relations", {}),
        "evidence_names": facts.get("evidence_names", []),
        "annotation_mapping": facts.get("annotation_mapping", []),
    }
    access = context.get("dataset_access_manifest", {})
    if access.get("schema_version") == "annotation.v5" and access.get("cases_path"):
        from medbenchagent.data.v5_index import V5DatasetQuery
        query = V5DatasetQuery(access)
        assets = {row["asset_id"]: row for row in query.assets()}
        node_context["case_facts"] = [case_facts(case, assets) for case in query.iter_cases()]
    nodes = build_task_nodes(plan.axis_analysis, node_context)
    active = sorted(key for key, node in nodes.items() if not node.exclusion_reason)
    if not active:
        raise ValueError("Axis analysis produced no executable Task Node")
    plan.nodes = nodes
    plan.phase = "leaves"
    plan.pending_node_ids = active
    plan.completed_node_ids = []
    plan.current_node_id = None
    return plan


def analyze_axis(context: dict[str, Any], *, axis: str, assessments: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "axes":
        raise ValueError(f"Axis analysis is not available in phase {plan.phase}")
    try:
        parsed_axis = TaskAxis(axis)
    except ValueError as exc:
        raise ValueError(f"Unknown Task Axis: {axis}") from exc
    expected = _axis_order()[len(plan.axis_assessments)]
    if parsed_axis != expected:
        raise ValueError(f"Expected Axis {expected}, received {parsed_axis}")
    parsed = [AxisAssessment.model_validate(value) for value in assessments]
    allowed = set(AXIS_VALUES[parsed_axis])
    actual = [value.value for value in parsed]
    if set(actual) != allowed or len(actual) != len(set(actual)):
        raise ValueError(f"Axis analysis must cover registry exactly for {parsed_axis}")
    plan.axis_assessments[parsed_axis.value] = parsed
    if len(plan.axis_assessments) == 3:
        plan.axis_analysis = AxisAnalysis(
            input_structure=plan.axis_assessments[TaskAxis.INPUT_STRUCTURE.value],
            operation=plan.axis_assessments[TaskAxis.OPERATION.value],
            clinical_target=plan.axis_assessments[TaskAxis.CLINICAL_TARGET.value],
            dataset_revision=str(plan.dataset_facts.get("revision") or context.get("dataset_access_manifest", {}).get("revision", "1")),
        )
        plan.next_axis = None
        plan = _advance_nodes(context, plan)
    else:
        plan.next_axis = _axis_order()[len(plan.axis_assessments)]
    updates = _write_derived_artifacts(context, plan)
    return result("success", f"Axis {parsed_axis} saved", **updates)


def analyze_axes(context: dict[str, Any], *, analysis: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper that imports a complete legacy analysis."""
    require(context, "Dataset", "PlanningConfig")
    parsed = AxisAnalysis.model_validate(analysis)
    plan = _load_plan(context)
    plan.axis_analysis = parsed
    plan.axis_assessments = {
        TaskAxis.INPUT_STRUCTURE.value: parsed.input_structure,
        TaskAxis.OPERATION.value: parsed.operation,
        TaskAxis.CLINICAL_TARGET.value: parsed.clinical_target,
    }
    plan.dataset_facts = {**plan.dataset_facts, **_fact_context(context)}
    plan.phase = "axes"
    plan = _advance_nodes(context, plan)
    versions = publish(context, "PlanningDraft", {"task_space_plan": str(_plan_path(context))}, ("Dataset", "PlanningConfig")) if context.get("run_directory") else {}
    return result("success", "Axis Analysis saved", **_write_derived_artifacts(context, plan), **versions)


def build_nodes(context: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper; normal planning builds Nodes after the third Axis."""
    plan = _load_plan(context)
    if plan.axis_analysis is None:
        path = context.get("axis_analysis_path")
        if not path:
            raise ValueError("Axis Analysis is required before building Nodes")
        plan.axis_analysis = AxisAnalysis.model_validate(_read(path))
    facts = _fact_context(context)
    plan.dataset_facts = {**plan.dataset_facts, **facts}
    plan = _advance_nodes(context, plan)
    return result("success", "Task Nodes saved", **_write_derived_artifacts(context, plan), exclusions={
        key: {"node": node.model_dump(mode="json"), "reason": node.exclusion_reason}
        for key, node in plan.nodes.items() if node.exclusion_reason
    })


def read_planning_context(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    return result("success", json.dumps({"facts": plan.dataset_facts, "blueprint": plan.blueprint}, ensure_ascii=False))


def read_axis_analysis(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    return result("success", json.dumps(plan.axis_analysis.model_dump(mode="json") if plan.axis_analysis else plan.axis_assessments, ensure_ascii=False))


def read_task_node(context: dict[str, Any], *, node_id: str | None = None) -> dict[str, Any]:
    plan = _load_plan(context)
    if node_id is None:
        node_id = plan.current_node_id or (plan.pending_node_ids[0] if plan.pending_node_ids else None)
    value = plan.nodes.get(node_id) if node_id else plan.nodes
    return result("success", json.dumps(value.model_dump(mode="json") if isinstance(value, TaskNode) else value, ensure_ascii=False))


def read_current_node_context(context: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves":
        raise ValueError(f"Node design is not available in phase {plan.phase}")
    if plan.current_node_id is None:
        if not plan.pending_node_ids:
            plan.phase = "capabilities"
            _write_derived_artifacts(context, plan)
            raise ValueError("All Nodes are complete; classify their Blueprint capabilities")
        plan.current_node_id = plan.pending_node_ids[0]
        node = plan.nodes[plan.current_node_id]
        plan.nodes[plan.current_node_id] = node.model_copy(update={"status": "active"})
        _write_derived_artifacts(context, plan)
    from medbenchagent.data.v5_index import V5DatasetQuery

    node = plan.nodes[plan.current_node_id]
    context_value = V5DatasetQuery(context["dataset_access_manifest"]).node_context(node)
    context_value["task_leaf_ids"] = node.leaf_ids
    context_value["plan_phase"] = plan.phase
    return result("success", json.dumps(context_value, ensure_ascii=False))


def node_context(context: dict[str, Any], *, node_id: str) -> dict[str, Any]:
    """Compatibility reader for a specific Node."""
    plan = _load_plan(context)
    node = plan.nodes.get(node_id)
    if node is None:
        raise ValueError(f"Unknown Task Node: {node_id}")
    from medbenchagent.data.v5_index import V5DatasetQuery
    return result("success", json.dumps(V5DatasetQuery(context["dataset_access_manifest"]).node_context(node), ensure_ascii=False))


def save_task_leaf(context: dict[str, Any], *, leaf: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves":
        raise ValueError("Task Leaf design is not available in the current planning phase")
    if plan.current_node_id is None:
        if not plan.pending_node_ids:
            raise ValueError("No pending Task Node is available")
        plan.current_node_id = plan.pending_node_ids[0]
        node = plan.nodes[plan.current_node_id]
        plan.nodes[plan.current_node_id] = node.model_copy(update={"status": "active"})
    parsed = TaskLeaf.model_validate(leaf)
    if parsed.node_id != plan.current_node_id:
        raise ValueError("Leaf must belong to the current active Node")
    evidence_names = set(plan.dataset_facts.get("evidence_names", []))
    evidence_names.update(
        selector.name for selector in parsed.input
        if selector.name and selector.evidence_type == "asset"
    )
    validate_candidate_leaf(parsed, plan.nodes, resolver_context={"evidence_names": evidence_names})
    identity = leaf_identity(parsed)
    existing = plan.leaves.get(parsed.leaf_id)
    if existing is not None and leaf_identity(existing) != identity:
        old_node = plan.nodes[existing.node_id]
        plan.nodes[existing.node_id] = old_node.model_copy(update={
            "leaf_ids": [value for value in old_node.leaf_ids if value != existing.leaf_id]
        })
    if any(key != parsed.leaf_id and leaf_identity(value) == identity for key, value in plan.leaves.items()):
        raise ValueError("Duplicate Task Leaf semantics")
    from medbenchagent.data.query import DatasetQuery

    query = DatasetQuery(context["dataset_access_manifest"])
    if hasattr(query, "iter_cases"):
        cases = list(query.iter_cases())
    else:
        cases, offset = [], 0
        while True:
            page = query.sample(offset=offset, limit=512)
            cases.extend(page.get("values", []))
            if page.get("complete", True):
                break
            offset = page["next_offset"]
    eligible = sum(not isinstance(resolve(case, ResolverSpec.parse(parsed.resolver)), NotApplicable) for case in cases)
    if eligible <= 0:
        raise ValueError("Task Leaf resolver does not resolve any Case")
    parsed = parsed.model_copy(update={"eligible_count": eligible})
    plan.leaves[parsed.leaf_id] = parsed
    node = plan.nodes[parsed.node_id]
    plan.nodes[parsed.node_id] = node.model_copy(update={"leaf_ids": [*node.leaf_ids, parsed.leaf_id], "status": "active"})
    return result("success", "Task Leaf saved", **_write_derived_artifacts(context, plan))


def review_current_node(context: dict[str, Any], *, review: dict[str, Any]) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves" or not plan.current_node_id:
        raise ValueError("A current active Node is required before review")
    parsed = NodeReview.model_validate(review)
    if parsed.node_id != plan.current_node_id:
        raise ValueError("Review must belong to the current active Node")
    node = plan.nodes[parsed.node_id]
    expected = set(node.leaf_ids)
    if set(parsed.checked_leaf_ids) != expected or not expected:
        raise ValueError("Node review must cover every saved Leaf")
    if parsed.status == "approved":
        if any((plan.leaves[leaf_id].eligible_count or 0) <= 0 for leaf_id in expected):
            raise ValueError("Cannot approve a Node with a zero-capacity Leaf")
        plan.nodes[parsed.node_id] = node.model_copy(update={"status": "completed"})
        plan.reviews[parsed.node_id] = parsed
        plan.completed_node_ids = sorted(set(plan.completed_node_ids) | {parsed.node_id})
        plan.pending_node_ids = [key for key in plan.pending_node_ids if key != parsed.node_id]
        plan.current_node_id = None
    else:
        plan.nodes[parsed.node_id] = node.model_copy(update={"status": "active"})
        plan.reviews[parsed.node_id] = parsed
    return result("success", f"Node {parsed.node_id} review saved", **_write_derived_artifacts(context, plan))


def classify_task_nodes(context: dict[str, Any], *, assignments: list[dict[str, Any]]) -> dict[str, Any]:
    plan = _load_plan(context)
    if plan.phase != "leaves" or plan.pending_node_ids or plan.current_node_id:
        raise ValueError("All Nodes must be reviewed before capability classification")
    blueprint = plan.blueprint or _load_blueprint_for_context(context)
    allowed = blueprint_dimension_ids(blueprint)
    parsed: dict[str, list[str]] = {}
    for assignment in assignments:
        node_id = assignment.get("node_id")
        dimensions = assignment.get("capability_dimension_ids")
        if node_id not in plan.nodes or plan.nodes[node_id].exclusion_reason:
            raise ValueError(f"Unknown executable Node in capability assignment: {node_id}")
        if not isinstance(dimensions, list) or not dimensions or not set(dimensions) <= allowed:
            raise ValueError(f"Invalid Blueprint capability dimensions for Node: {node_id}")
        parsed[node_id] = list(dict.fromkeys(dimensions))
    expected = {key for key, node in plan.nodes.items() if not node.exclusion_reason}
    if set(parsed) != expected:
        raise ValueError("Capability assignments must cover every executable Node exactly once")
    plan.capability_assignments = parsed
    plan.nodes = {
        key: node.model_copy(update={"capability_dimension_ids": parsed[key]})
        for key, node in plan.nodes.items()
    }
    plan.leaves = {
        key: leaf.model_copy(update={"capability_dimension_ids": parsed.get(leaf.node_id, [])})
        for key, leaf in plan.leaves.items()
    }
    plan.phase = "complete"
    return result("success", "Blueprint capabilities assigned", **_write_derived_artifacts(context, plan))


def finish_task_space(context: dict[str, Any]) -> dict[str, Any]:
    """Publish the complete validated Task Space artifact."""
    require(context, "Dataset", "PlanningConfig")
    plan = _load_plan(context)
    validate_task_tree(plan, require_capabilities=True, require_reviews=True)
    plan.phase = "complete"
    updates = _write_derived_artifacts(context, plan)
    versions = publish(context, "Plan", {"task_space_plan": updates["task_space_plan_path"]}, ("Dataset", "PlanningConfig")) if context.get("run_directory") else {}
    return result("complete", "Task Space plan published", working_blueprint_path=updates["task_space_plan_path"], **updates, **versions)


def _available_tools(context: dict[str, Any]) -> set[str]:
    plan = _load_plan(context)
    if plan.phase == "blocked":
        return {"read_planning_context"}
    if plan.phase == "facts":
        return {"read_planning_context"}
    if plan.phase == "axes":
        return {"read_planning_context", "read_axis_analysis", "analyze_axis"}
    if plan.phase == "leaves":
        return {"read_current_node_context", "save_task_leaf", "review_current_node", "read_task_node"}
    if plan.phase == "capabilities":
        return {"classify_task_nodes", "read_task_node"}
    if plan.phase == "complete":
        return {"finish"}
    return set()


def compile_task_planning_agent(*, checkpointer=None):
    funcs = (
        read_planning_context,
        analyze_axis,
        read_axis_analysis,
        read_current_node_context,
        read_task_node,
        save_task_leaf,
        review_current_node,
        classify_task_nodes,
    )
    return compile_agent(
        "TaskPlanningAgent",
        "Build one resumable annotation.v5 Task Plan. The deterministic planning context is available before the first decision. Call analyze_axis exactly three times in order: input_structure, operation, clinical_target. Then process the current Node only, save at least one evidence-backed Leaf, review all Leaves for that Node, and classify every completed Node with one or more registered Blueprint capability dimensions. Never invent labels, skip Nodes, or use reference evidence as model input.",
        {f.__name__: f for f in funcs},
        lambda c: {
            "task_space_plan": c.get("task_space_plan_path"),
            "phase": _load_plan(c).phase,
            "current_node_id": _load_plan(c).current_node_id,
            "pending_node_count": len(_load_plan(c).pending_node_ids),
            "planning_context": _load_plan(c).dataset_facts,
        },
        finish_task_space,
        args_schemas={
            "analyze_axis": AxisArguments,
            "read_task_node": TaskNodeArguments,
            "save_task_leaf": TaskLeafArguments,
            "review_current_node": NodeReviewArguments,
            "classify_task_nodes": CapabilityAssignmentArguments,
        },
        available_tools=_available_tools,
        initialize_context=_initialize_planning_context,
        checkpointer=checkpointer,
    )


def apply_template_feedback(context, feedback):
    return result("blocked", "Template review is part of Node planning")
