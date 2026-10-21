"""Artifact-driven supervision and bounded, scoped requests for planning rework."""
from __future__ import annotations

from collections import Counter
from pydantic import Field

from medbenchagent.planning.bir import BIRStore, business_dependencies
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty


class RevisionArguments(NoArguments):
    scenario_ids: list[NonEmpty] = Field(min_length=1)
    reason: NonEmpty
    issue_ids: list[NonEmpty] = Field(default_factory=list)


class PublishArguments(NoArguments):
    limitation_reason: str = ""


def current_document(context, path):
    try:
        return BIRStore(context["bir_path"]).load(path)
    except (OSError, ValueError, KeyError):
        return None


def compilation_summary(context):
    selected = context.get("selected_dataset_ids", [])
    dataset = selected[0] if len(selected) == 1 else None
    plan = current_document(context, f"datasets/{dataset}/planning/task_plan.json") if dataset else None
    review = current_document(context, "review.json")
    candidate = current_document(context, "candidate.json")
    return {
        "compilation_id": context.get("compilation_id"), "dataset_id": dataset,
        "requirements": context.get("benchmark_info", {}),
        "plan": {"phase": plan.get("phase"), "revision": plan.get("revision"),
                 "leaf_count": len(plan.get("leaf_tasks", {})),
                 "capacity": sum(row.get("eligible_count") or 0 for row in plan.get("leaf_tasks", {}).values()),
                 "scenarios": [{"scenario_id": key, "status": row["status"],
                                "exclusion_reason": row.get("exclusion_reason"),
                                "exclusion_category": row.get("exclusion_category"),
                                "leaf_count": len(row.get("leaf_task_ids", []))}
                               for key, row in plan.get("scenarios", {}).items()]} if plan else None,
        "review": review,
        "candidate": {key: candidate.get(key) for key in ("counts", "status")} if candidate else None,
        "last_agent_result": context.get("last_agent_result"),
        "delivery_path": context.get("delivery_path"),
    }


def inspect_compilation(context):
    """Inspect current coverage, review findings and audited candidate quality before deciding."""
    import json
    return result("success", json.dumps(compilation_summary(context), ensure_ascii=False))


def compilation_view(context):
    summary = compilation_summary(context)
    plan, review = summary.get("plan") or {}, summary.get("review") or {}
    return {"dataset_id": summary["dataset_id"], "compilation_id": summary["compilation_id"],
            "plan": {key: plan[key] for key in ("phase", "leaf_count", "capacity") if key in plan},
            "review": {"status": review.get("status"), "finding_count": len(review.get("findings", []))},
            "candidate": summary["candidate"], "delivery_path": summary["delivery_path"]}


def revise_plan(context, *, scenario_ids, reason, issue_ids=None):
    """Reopen specific Scenarios for evidence-based repair or additional exploration."""
    store = BIRStore(context["bir_path"])
    selected = list(store.load("dataset_bindings.json")["datasets"])
    if len(selected) != 1:
        raise ValueError("Planning revision requires one dataset")
    dataset = selected[0]
    path = f"datasets/{dataset}/planning/task_plan.json"
    plan = TaskPlanState.model_validate(store.load(path))
    targets = list(dict.fromkeys(scenario_ids))
    if not targets or not set(targets) <= set(plan.scenarios):
        raise ValueError("Revision must name existing Scenarios")
    review = current_document(context, "review.json") or {}
    known_issues = {row["id"]: row for row in review.get("findings", [])}
    if any(key not in known_issues or known_issues[key]["scenario_id"] not in targets
           for key in (issue_ids or [])):
        raise ValueError("Revision issues must belong to the current review and selected Scenarios")
    journal = {"revisions": plan.review.get("revision_history", [])}
    counts = Counter(key for entry in journal["revisions"] for key in entry["scenario_ids"])
    if any(counts[key] >= context.get("limits", {}).get("max_reworks", 3) for key in targets):
        raise ValueError("Scenario rework budget exhausted; report the unresolved limitation")
    if any(plan.scenarios[key].exclusion_stage == "structural" for key in targets):
        raise ValueError("Structural infeasibility requires new data evidence, not a planning override")
    before = store.digest(path)
    for key in targets:
        scenario = plan.scenarios[key]
        plan.scenarios[key] = scenario.model_copy(update={
            "status": "active", "exclusion_reason": None, "exclusion_stage": None,
            "exclusion_category": None, "exclusion_evidence": [],
        })
        plan.reviews.pop(key, None)
    plan.completed_scenario_ids = [key for key in plan.completed_scenario_ids if key not in targets]
    screening = [key for key in targets if not plan.scenarios[key].validated_clinical_targets]
    pending = list(dict.fromkeys([*targets, *plan.pending_scenario_ids]))
    plan.pending_scenario_ids = screening or pending
    plan.phase = "screening" if screening else "leaves"
    plan.current_scenario_id = (screening or pending)[0]
    plan.revision += 1
    journal["revisions"].append({"scenario_ids": targets, "reason": reason, "issue_ids": issue_ids or [],
                                 "input_digest": before, "plan_revision": plan.revision})
    plan.review = {"revision_request": {"scenario_ids": targets, "reason": reason, "issue_ids": issue_ids or [],
                                        "findings": [known_issues[key] for key in (issue_ids or [])]},
                   "revision_history": journal["revisions"]}
    store.update(path, plan.model_dump(mode="json"), run_id=context["run_id"], actor="supervisor_revision",
                 expected_digest=before,
                 dependencies=business_dependencies(context, "benchmark_info.json", "dataset_bindings.json", registry=True))
    return result("success", "Selected Scenarios reopened; downstream receipts are now stale",
                  planning_stalled=False, planning_feedback=plan.review["revision_request"],
                  dataset_task_plans={}, allocation_path=None, allocation=None,
                  dataset_generations={}, delivery_path=None, candidate_path=None)
