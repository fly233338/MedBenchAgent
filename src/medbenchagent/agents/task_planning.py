"""Task Space planning Agent for annotation.v5 datasets."""

from pathlib import Path
import json
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.tool_contracts import NoArguments
from medbenchagent.runtime.dependencies import require, publish
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.planning.task_space import AxisAnalysis, TaskLeaf, TaskNode
from medbenchagent.planning.node_builder import build_task_nodes
from medbenchagent.planning.task_discovery import validate_candidate_leaf
from medbenchagent.planning.resolver import resolve, ResolverSpec, NotApplicable


class AxisAnalysisArguments(NoArguments):
    analysis: dict


class TaskNodeArguments(NoArguments):
    node_id: str | None = None


class TaskLeafArguments(NoArguments):
    leaf: dict


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nodes(context):
    return _read(context["task_nodes_path"])


def _profile(context):
    return _read(context["dataset_access_manifest"]["profile_path"])


def analyze_axes(context, *, analysis):
    require(context, "Dataset", "PlanningConfig")
    parsed = AxisAnalysis.model_validate(analysis)
    path = Path(context["operation_directory"]) / "axis_analysis.json"
    atomic_write_json(path, parsed.model_dump())
    versions = publish(context, "PlanningDraft", {"axis_analysis": str(path)}, ("Dataset", "PlanningConfig")) if context.get("run_directory") else {}
    return result("success", "Axis Analysis saved", axis_analysis_path=str(path), **versions)


def build_nodes(context):
    profile = _profile(context)
    analysis = AxisAnalysis.model_validate(_read(context["axis_analysis_path"]))
    evidence = {}
    for row in profile.get("evidence", []):
        for kind, count in row.get("types", {}).items():
            evidence[kind] = evidence.get(kind, 0) + count
        evidence["numeric"] = evidence.get("numeric", 0) + row.get("value_types", {}).get(
            "number", 0
        )
    assets = profile.get("assets", {})
    per = {int(k): v for k, v in assets.get("assets_per_case", {}).items()}
    related = sum(v for k, v in per.items() if k >= 2)
    facts = {
        "evidence": evidence,
        "assets": {
            "volume_assets": assets.get("kinds", {}).get("volume", 0),
            "image_assets": assets.get("kinds", {}).get("image", 0),
            "related_image_cases": related,
            "multi_image_cases": related,
        },
        "relations": {
            "related": sum(r.get("has_ref", 0) for r in profile.get("evidence", [])),
            "paired": related,
            "paired_study": related,
            "paired_text": sum(
                r.get("types", {}).get("text", 0) for r in profile.get("evidence", [])
            ),
        },
    }
    node_context_path = Path(context["operation_directory"]) / "node_context.json"
    node_context = {"evidence": {row["name"]: row for row in profile.get("evidence", [])},
                    "evidence_profile": profile.get("evidence", []),
                    "assets": facts["assets"], "relations": facts["relations"]}
    atomic_write_json(node_context_path, node_context)
    exclusions = {}
    nodes = build_task_nodes(analysis, facts, exclusions=exclusions)
    path = Path(context["operation_directory"]) / "task_nodes.json"
    atomic_write_json(path, {k: v.model_dump() for k, v in nodes.items()})
    return result(
        "success",
        "Task Nodes saved",
        task_nodes_path=str(path),
        task_nodes={k: v.model_dump() for k, v in nodes.items()},
        node_context_path=str(node_context_path), node_context=node_context,
        exclusions=exclusions,
    )


def read_axis_analysis(context):
    return result("success", json.dumps(_read(context["axis_analysis_path"]), ensure_ascii=False))


def read_task_node(context, *, node_id=None):
    nodes = _nodes(context)
    return result("success", json.dumps(nodes.get(node_id, nodes), ensure_ascii=False))


def node_context(context, *, node_id):
    from medbenchagent.data.v5_index import V5DatasetQuery

    node = TaskNode.model_validate(_nodes(context)[node_id])
    return result(
        "success",
        json.dumps(
            V5DatasetQuery(context["dataset_access_manifest"]).node_context(node),
            ensure_ascii=False,
        ),
    )


def save_task_leaf(context, *, leaf):
    parsed = TaskLeaf.model_validate(leaf)
    profile = _profile(context)
    evidence_names = {row["name"] for row in profile.get("evidence", [])}
    evidence_names.update(selector.name for selector in parsed.input
                          if selector.name and selector.evidence_type == "asset")
    validate_candidate_leaf(
        parsed, {k: TaskNode.model_validate(v) for k, v in _nodes(context).items()},
        resolver_context={"evidence_names": evidence_names},
    )
    from medbenchagent.data.query import DatasetQuery

    query = DatasetQuery(context["dataset_access_manifest"])
    cases = []
    offset = 0
    while True:
        page = query.sample(offset=offset, limit=512)
        cases.extend(page["values"])
        if page.get("complete", True):
            break
        offset = page["next_offset"]
    eligible = sum(
        not isinstance(resolve(case, ResolverSpec.parse(parsed.resolver)), NotApplicable)
        for case in cases
    )
    parsed = parsed.model_copy(update={"eligible_count": eligible})
    path = Path(context["operation_directory"]) / "task_leaves.json"
    leaves = _read(path) if path.is_file() else {}
    leaves[parsed.leaf_id] = parsed.model_dump()
    atomic_write_json(path, leaves)
    return result("success", "Task Leaf saved", task_leaves_path=str(path), task_leaves=leaves)


def finish_task_space(context):
    """Publish the validated v2 Task Space artifact after all required inputs exist."""
    require(context, "Dataset", "PlanningConfig")
    for key in ("axis_analysis_path", "task_nodes_path", "task_leaves_path"):
        if not context.get(key) or not Path(context[key]).is_file():
            raise ValueError(f"{key} is required before publishing the Task Space")
    path = Path(context["operation_directory"]) / "task_space_plan.json"
    atomic_write_json(path, {"schema_version": "task-space-plan.v1",
                             "dataset": {"dataset_id": context.get("dataset_scope")},
                             "scope": {"dataset_id": context.get("dataset_scope")},
                             "revision": _read(context["axis_analysis_path"]).get("dataset_revision", "1"),
                             "axis_analysis": _read(context["axis_analysis_path"]),
                             "nodes": _read(context["task_nodes_path"]),
                             "leaves": _read(context["task_leaves_path"]), "review": {}})
    versions = publish(context, "Plan", {"blueprint": str(path)}, ("Dataset", "PlanningConfig"))
    return result("complete", "Task Space plan published", working_blueprint_path=str(path), **versions)


def compile_task_planning_agent(*, checkpointer=None):
    funcs = (
        analyze_axes,
        read_axis_analysis,
        build_nodes,
        read_task_node,
        node_context,
        save_task_leaf,
    )
    return compile_agent(
        "TaskPlanningAgent",
        "Build the annotation.v5 Task Space. First analyze only the fixed Input Structure, Operation and Clinical Target axes; then inspect each feasible Node and discover only evidence-backed Leaves. Never invent labels, propagate Case-level evidence, choose metrics, or use reference evidence as model input.",
        {f.__name__: f for f in funcs},
        lambda c: {
            "task_nodes": c.get("task_nodes_path"),
            "task_leaves": c.get("task_leaves_path"),
        },
        finish_task_space,
        args_schemas={
            "analyze_axes": AxisAnalysisArguments,
            "read_task_node": TaskNodeArguments,
            "node_context": TaskNodeArguments,
            "save_task_leaf": TaskLeafArguments,
        },
        available_tools=lambda c: {f.__name__ for f in funcs},
        checkpointer=checkpointer,
    )


def apply_template_feedback(context, feedback):
    return result("blocked", "Template stage is removed from v2")
