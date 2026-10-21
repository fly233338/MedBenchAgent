"""Independent semantic review, without permission to mutate a Task Plan."""
import json
from medbenchagent.agents.task_planning import (
    _load_plan, _screening_payload, _stage_prompt, _validate_screening_targets,
    evaluate_scenario, ScenarioScreeningResult,
)
from medbenchagent.planning.bir import BIRStore, business_dependencies
from medbenchagent.runtime.harness import result
from medbenchagent.planning.context import complete_planning_json as complete_json


def review_plan(context):
    store = BIRStore(context["bir_path"])
    plan = _load_plan(context)
    if not plan.scenarios:
        raise ValueError("Independent review requires discovered Scenarios")
    dependencies = business_dependencies(context, "task_plan.json",
                                         "dataset_binding.json", "benchmark_info.json", registry=True)
    saved = store.current("review.json") or {"schema_version": "plan-review", "findings": [], "checked_scenario_ids": [], "status": "running"}
    for key, scenario in plan.scenarios.items():
        if key in saved["checked_scenario_ids"]:
            continue
        findings = []
        if scenario.leaf_task_ids:
            review = evaluate_scenario(dict(context), plan, scenario)
            if review.scenario_id != key or set(review.checked_leaf_task_ids) != set(scenario.leaf_task_ids):
                raise ValueError("Independent review must cover the current Scenario and every Leaf")
            issues = [*review.issues, *(f"{row.leaf_task_id}: {row.reason}" for row in review.discarded_leaf_tasks)]
            blocking = review.status != "approved" or bool(review.discarded_leaf_tasks)
            if blocking and not issues:
                issues = ["Reviewer rejected the Scenario; task meaning or evidence needs repair"]
            findings = [{"id": f"{key}:{index}", "scenario_id": key, "owner": "planning",
                         "category": "invalid_task" if blocking else "coverage_gap", "blocking": blocking,
                         "summary": text, "evidence_refs": [f"scenario:{key}"]}
                        for index, text in enumerate(issues)]
            from medbenchagent.agents.task_planning import _facts
            known_fields = set(_facts(context)["evidence_catalog"])
            for index, concern in enumerate(review.source_concerns):
                if not set(concern.fields) <= known_fields:
                    raise ValueError("Source concern references unknown evidence fields")
                findings.append({"id": f"{key}:source:{index}", "scenario_id": key, "owner": "preparation",
                                 "category": "source_semantics", "blocking": True, "summary": concern.summary,
                                 "fields": concern.fields, "evidence_refs": [item.reference for item in concern.evidence_refs]})
        elif scenario.exclusion_reason and scenario.exclusion_stage != "structural":
            raw = complete_json(_stage_prompt(context, "scenario_screening"),
                                json.dumps(_screening_payload(context, plan, scenario), ensure_ascii=False),
                                ScenarioScreeningResult,
                                validate=lambda value: _validate_screening_targets(context, plan, scenario, value.validated_clinical_targets))
            decision = ScenarioScreeningResult.model_validate(raw)
            if decision.decision == "keep":
                findings = [{"id": f"{key}:coverage", "scenario_id": key, "owner": "planning",
                             "category": "coverage_gap", "blocking": True, "summary": decision.reason,
                             "evidence_refs": [item.reference for item in decision.evidence_basis]}]
        saved["findings"].extend(findings)
        saved["checked_scenario_ids"].append(key)
        saved["status"] = "running"
        store.update("review.json", saved, run_id=context["run_id"], actor="independent_review", dependencies=dependencies)
    saved["status"] = "rejected" if any(row["blocking"] for row in saved["findings"]) else "approved"
    store.update("review.json", saved, run_id=context["run_id"], actor="independent_review", dependencies=dependencies)
    return result("success", json.dumps(saved, ensure_ascii=False))
