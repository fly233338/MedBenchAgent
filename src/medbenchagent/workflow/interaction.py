"""Domain clarification adapter for the generic agent executor."""
from langgraph.types import interrupt
from medbenchagent.runtime.construction import event_context
from medbenchagent.runtime.harness import result
from medbenchagent.workflow.clarifications import (
    new_issue, answer_payload, record_answer, resolve_semantic_answer,
)


def ask_user(context, name, arguments, operation):
    issue = None
    try:
        question = arguments["question"]
        reference = arguments["reference"] or "clarification"
        issue = context.get("global_issue") if name == "SupervisorAgent" else context.get("issues", {}).get(arguments["issue_id"]) if arguments["issue_id"] else None
        if issue and name != "SupervisorAgent" and issue["role"] != name:
            raise ValueError("Cannot answer another agent's issue")
        if issue and arguments["issue_id"] != issue["issue_id"]:
            outcome = result("failed", "ask_user must reference the pending issue_id")
            issue = None
            invalid_issue = True
        else:
            invalid_issue = bool(arguments["issue_id"] and not issue)
        issue = issue or new_issue(name, context, reference, question, subject_scope=arguments["subject_scope"],
                                   source_ids=arguments["source_ids"])
        if name == "SupervisorAgent" and not issue.get("purpose"):
            metadata = {
                key: arguments[key]
                for key in ("purpose", "candidate_snapshot_path", "allowed_dataset_ids", "allowed_extension_ids")
                if arguments.get(key)
            }
            if metadata:
                issue = {**issue, **metadata}
        decision_key = issue["issue_id"]
        issue = context.get("issues", {}).get(decision_key, issue)
        decisions = dict(context.get("decisions", {}))
        if invalid_issue:
            outcome = result("failed", "ask_user must reference the pending issue_id, or omit it for a new question")
        elif issue["status"] == "resolved":
            outcome = result("success", "Question already resolved", global_issue=None,
                             decisions={**decisions, decision_key: issue["answer_reference"]}, clarification=issue)
        elif arguments["scope"] == "global" and name != "SupervisorAgent":
            outcome = result(
                "needs_input",
                question,
                unresolved_questions=[question],
                global_issue=issue,
                issues={**context.get("issues", {}), decision_key: issue},
            )
        else:
            answer = answer_payload(context, issue)["text"] if issue.get("answer_reference") else None
            followup = issue["status"] == "answered" and question != issue["question"]
            if followup:
                answer = None
            if answer is None:
                answer = interrupt(
                    {
                        "kind": "clarification",
                        "role": name,
                        "question": question if followup else issue["question"],
                        "reference": issue["reference"],
                        "issue_id": decision_key,
                    }
                )
            if not isinstance(answer, str) or not answer.strip():
                outcome = result("failed", "A natural-language answer is required")
            else:
                if answer is not None and (followup or not issue.get("answer_reference")):
                    issue = record_answer(context, issue, answer, question=question if followup else None)
                if followup:
                    issue = {**issue, "question": question}
                semantic_error = None
                if name != "SupervisorAgent":
                    try:
                        with event_context(construction_events_path=context["construction_events_path"], run_id=context["run_id"],
                         agent=name, dataset_id=context.get("dataset_scope"), source_id=context.get("source_scope"), operation="resolve_semantic"):
                            issue = resolve_semantic_answer(context, issue)
                        decisions[decision_key] = issue["answer_reference"]
                    except ValueError as exc:
                        semantic_error = str(exc)
                updates = {
                    "decisions": decisions,
                    "clarification": issue,
                    "issues": {**context.get("issues", {}), decision_key: issue},
                    "global_issue": issue if name == "SupervisorAgent" else None,
                }
                if name == "SupervisorAgent" and issue.get("purpose") == "benchmark_fields":
                    from medbenchagent.workflow.clarifications import resolve_clarification
                    completion = resolve_clarification(
                        {**context, **updates, "operation_directory": str(operation.with_suffix(""))},
                        issue_id=decision_key, kind="requirement_change",
                    )
                    updates.update(completion["updates"])
                outcome = result(
                    "failed" if semantic_error else "success",
                    "User clarification received" if not semantic_error else semantic_error,
                    **updates,
                )
    except ValueError as exc:
        saved_issue = {"issues": {**context.get("issues", {}), issue["issue_id"]: issue}} if issue and issue.get("answer_reference") else {}
        outcome = result("failed", str(exc), **saved_issue)

    return outcome
