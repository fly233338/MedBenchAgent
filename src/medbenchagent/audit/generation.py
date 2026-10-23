"""Deterministic auditing of the new prepared-data generation contract."""

import json

from medbenchagent.generation.backends import assemble_item
from medbenchagent.generation.contracts import AuditReport, validate_generation_spec
from medbenchagent.generation.execution import selected_cases


def audit_candidate(candidate, target, executions):
    """executions maps spec artifact IDs to (binding, spec, plan, TaskDataReader)."""
    candidate.verify_integrity()
    findings = []
    tasks = {}
    for expected in candidate.manifest.specs:
        actual, spec, plan, reader = executions[expected.artifact_id]
        if actual != expected:
            raise ValueError("Candidate specification identity mismatch")
        validate_generation_spec(
            spec,
            plan=plan,
            plan_artifact_id=spec.task_plan.artifact_id,
            plan_digest=spec.task_plan.digest,
            reader=reader,
        )
        if spec.task_id in tasks:
            raise ValueError("Candidate has multiple specifications for the same task")
        tasks[spec.task_id] = (spec, plan.tasks[spec.task_id], reader)
    counts = {}
    source_keys = set()
    contents = set()
    for item in candidate.items:
        try:
            spec, task, reader = tasks[item.leaf_task_id]
            if (
                item.case_id
                not in selected_cases(spec, task, reader)[
                    : candidate.manifest.allocation.counts[task.task_id]
                ]
            ):
                raise ValueError("Case is outside the generation selection")
            key = (item.leaf_task_id, item.case_id)
            if key in source_keys:
                raise ValueError("Duplicate task/Case instance")
            source_keys.add(key)
            expected = assemble_item(task, reader, item.case_id, spec.seed)
            if item != expected:
                raise ValueError(
                    "Question, inputs, reference or provenance differs from the Leaf assembly"
                )
            content = json.dumps(
                {
                    "task": item.leaf_task_id,
                    "question": item.question,
                    "answer": item.answer,
                    "inputs": [
                        {
                            "kind": entry.kind,
                            "role": entry.role,
                            "value": entry.value["file_digest"]
                            if entry.kind == "asset"
                            else entry.value,
                        }
                        for entry in item.inputs
                    ],
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if content in contents:
                raise ValueError("Duplicate question content within the same task")
            contents.add(content)
            counts[item.leaf_task_id] = counts.get(item.leaf_task_id, 0) + 1
        except (ValueError, KeyError) as exc:
            findings.append(
                {
                    "item_id": item.item_id,
                    "task_id": item.leaf_task_id,
                    "code": "item_mismatch",
                    "message": str(exc),
                    "blocking": True,
                }
            )
    for key, quota in candidate.manifest.allocation.counts.items():
        if counts.get(key, 0) > quota:
            findings.append({"task_id": key, "code": "quota_exceeded", "blocking": True})
    return AuditReport(
        target=target,
        passed=not findings,
        checked_count=len(candidate.items),
        findings=findings,
        counts=dict(candidate.manifest.counts),
    )
