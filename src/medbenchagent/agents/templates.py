"""Persistent template review after Planning, before Allocation."""

import json
from pathlib import Path

from medbenchagent.generation.templates import create_templates, review_templates
from medbenchagent.planning.protocol import template_specification
from medbenchagent.runtime.construction import event_context
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


def prepare_templates(context):
    """Approve every executable Leaf's templates, or return precise Planning revision requests."""
    versions = require(context, "Plan", "TemplateConfig")
    plan = json.loads(Path(context["working_blueprint_path"]).read_text(encoding="utf-8"))
    root = (Path(context["run_directory"]) / "agents" / "TaskPlanningAgent" /
            context["dataset_scope"] / "templates")
    ledger_path = root / "attempts.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
    templates, reviews, issues = {}, {}, []
    for leaf_id, leaf in plan["tasks"].items():
        specification = template_specification(leaf["protocol"], plan["benchmark_info"]["language"])
        key = {"specification": specification.model_dump(), "config": versions["TemplateConfig"]["id"]}
        attempts = ledger.setdefault(leaf_id, [])
        cached = next((a for a in reversed(attempts) if a["input"] == key and a.get("review", {}).get("verdict") == "approved"), None)
        if cached is None:
            while True:
                last = next((a for a in reversed(attempts) if a["input"] == key), None)
                if last and last.get("review", {}).get("verdict") == "revise_plan":
                    issues.append({"leaf_id": leaf_id, "reason": last["review"]["reason"],
                                   "specification": last["input"]["specification"], "input": last["input"],
                                   "templates": last["templates"],
                                   "feedback_id": f"{context['dataset_scope']}:{leaf_id}:{attempts.index(last)}"})
                    break
                # A saved generated set can resume its review without regenerating it.
                if last and "review" not in last:
                    attempt = last
                else:
                    if len(attempts) >= 1 + context.get("limits", {}).get("max_reworks", 3):
                        return result("blocked", f"Template rework budget exhausted for {leaf_id}", template_ledger_path=str(ledger_path))
                    attempt = {"input": key, "feedback": last.get("review", {}).get("reason", "") if last else ""}
                    attempts.append(attempt)
                    atomic_write_json(ledger_path, ledger)
                if "templates" not in attempt:
                    try:
                        with event_context(agent="TaskPlanningAgent", task_id=leaf_id, operation="create_templates"):
                            attempt["templates"] = create_templates(specification, attempt["feedback"],
                                temperature=context.get("question_temperature", get_settings().generation.question_temperature))
                    except ValueError as exc:
                        attempt["review"] = {"verdict": "revise_templates", "reason": str(exc)}
                    atomic_write_json(ledger_path, ledger)
                    if "templates" not in attempt:
                        continue
                with event_context(agent="TaskPlanningAgent", task_id=leaf_id, operation="review_templates"):
                    attempt["review"] = review_templates(specification, attempt["templates"], attempt["feedback"])
                atomic_write_json(ledger_path, ledger)
                if attempt["review"]["verdict"] == "approved":
                    cached = attempt
                    break
        if cached:
            templates[leaf_id] = cached["templates"]
            reviews[leaf_id] = {**key, "review": cached["review"]}
    if issues:
        return result("success", "Template review requires Planning revision", template_revision_request={
            "plan_version": versions["Plan"]["id"], "issues": issues,
            "feedback_id": json.dumps(sorted(i["feedback_id"] for i in issues))})
    path = Path(context["operation_directory"]) / "templates.json"
    atomic_write_json(path, {"templates": templates, "reviews": reviews})
    return result("success", "All task templates approved", templates_path=str(path), template_revision_request=None,
                  **publish(context, "Templates", {"path": str(path)}))
