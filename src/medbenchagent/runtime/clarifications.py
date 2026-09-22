"""Correlated clarification records; answers travel only to their originating agent."""

import json

from medbenchagent.runtime.tool_contracts import NoArguments, NonEmpty
from typing import Literal


class ResolveClarificationArguments(NoArguments):
    issue_id: NonEmpty
    kind: Literal["semantic", "requirement_change"]


def new_issue(role, context, reference, question):
    dataset = context.get("dataset_scope")
    source = context.get("source_scope")
    identity = json.dumps([role, dataset, source, reference, question], ensure_ascii=False)
    return {"issue_id": identity, "role": role, "dataset_id": dataset,
            "source_id": source, "input_versions": dict(context.get("artifact_versions", {})),
            "reference": reference, "question": question, "status": "pending"}


def inject_answers(parent, child, role, dataset=None):
    """Project resolved answers without copying another agent's private context."""
    issues = dict(child.get("issues", {}))
    decisions = dict(child.get("decisions", {}))
    for key, issue in parent.get("issues", {}).items():
        if issue["role"] == role and issue["dataset_id"] == dataset and issue["status"] == "resolved":
            issues[key] = issue
            decisions[key] = issue["answer"]
            child["clarification"] = issue
    child.update(issues=issues, decisions=decisions)
    pending = child.get("global_issue")
    if pending and issues.get(pending["issue_id"], {}).get("status") == "resolved":
        child["global_issue"] = None
        child["unresolved_questions"] = []
    return child


def resolve_clarification(context, *, issue_id, kind):
    """Classify the user's received answer: source semantics or an explicit benchmark requirement change."""
    from medbenchagent.runtime.harness import result
    issue = context.get("issues", {}).get(issue_id)
    if not issue or issue["status"] not in {"answered", "resolved"}:
        raise ValueError("Receive an answer for this issue before resolving it")
    if issue["status"] == "resolved":
        if issue["kind"] != kind:
            raise ValueError("This issue already has a different resolution")
        return result("success", "Clarification already resolved")
    issue = {**issue, "kind": kind, "status": "resolved"}
    updates = {"issues": {**context.get("issues", {}), issue_id: issue},
               "decisions": {**context.get("decisions", {}), issue_id: issue["answer"]},
               "global_issue": None, "clarification": issue}
    if kind == "requirement_change":
        from medbenchagent.runtime.dependencies import publish
        exchanges = issue.get("exchanges", [{"question": issue["question"], "answer": issue["answer"]}])
        clarification_text = "\n".join(f"Clarification question: {e['question']}\nUser answer: {e['answer']}" for e in exchanges)
        updates.update(user_requirement=context["user_requirement"] + "\n" + clarification_text,
                       user_inputs=[*context.get("user_inputs", [context["user_requirement"]]),
                                    *(e["answer"] for e in exchanges)],
                       brief_ready=False, brief_missing_fields=[])
        updates.update(publish(context, "Requirement", {"text": updates["user_requirement"]}, ()))
    from pathlib import Path
    from medbenchagent.runtime.run_artifacts import atomic_write_json
    # Replace the saved private-state reference too; re-entry is not required to
    # close the issue, and an interrupted resolution can safely repeat this write.
    states = {d: dict(roles) for d, roles in context.get("child_states", {}).items()}
    role = {"TaskPlanningAgent": "planning", "DataProcessingAgent": "preparation"}.get(issue["role"])
    field = {"AllocationAgent": "allocation_state_path", "FinalizationAgent": "finalization_state_path"}.get(issue["role"])
    saved = states.get(issue["dataset_id"], {}).get(role) if role else context.get(field) if field else None
    if saved:
        child = json.loads(Path(saved).read_text(encoding="utf-8"))
        inject_answers(updates, child, issue["role"], issue["dataset_id"])
        target = Path(context["operation_directory"]) / "resolved-child.json"
        atomic_write_json(target, child)
        if role:
            states[issue["dataset_id"]][role] = str(target)
            updates["child_states"] = states
        else:
            updates[field] = str(target)
    return result("success", "Clarification resolved and routed to its origin", **updates)
