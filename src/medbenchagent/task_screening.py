"""LLM-assisted task support screening against a deterministic dataset profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medbenchagent.llm import complete_json


class TaskScreeningReviewRequired(ValueError):
    """The model twice returned a task decision that references invalid profile paths."""

    def __init__(self, task_id: str, response: dict[str, Any], valid_paths: list[str]) -> None:
        self.task_id = task_id
        self.response = response
        self.valid_paths = valid_paths
        super().__init__(f"Task {task_id} requires a user-reviewed support decision.")


def _screening_prompt(task_id: str, task: dict[str, Any], profile: dict[str, Any]) -> str:
    return (
        "Decide whether one predefined medical VLM benchmark task is supported by an image-level "
        "annotation dataset. Return JSON only with task_id, decision, supporting_fields, reason. "
        "decision must be retain or drop. supporting_fields must contain only annotation paths that "
        "exist in dataset_profile.fields. A drop decision must have an empty supporting_fields list. "
        "Do not invent fields or infer fields from their names.\n\n"
        f"Task ID: {task_id}\n"
        f"Task: {json.dumps(task, ensure_ascii=False)}\n\n"
        f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}"
    )


def _normalize_decision(
    task_id: str, response: dict[str, Any], valid_paths: set[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    decision = response.get("decision")
    supporting_fields = response.get("supporting_fields")
    reason = response.get("reason")
    if response.get("task_id") != task_id or decision not in {"retain", "drop"}:
        return None, ["task_id or decision is invalid"]
    if not isinstance(supporting_fields, list) or not all(isinstance(field, str) for field in supporting_fields):
        return None, ["supporting_fields must be a list of strings"]
    if not isinstance(reason, str) or not reason.strip():
        return None, ["reason is required"]
    invalid_paths = sorted(set(supporting_fields) - valid_paths)
    if decision == "drop" and supporting_fields:
        invalid_paths.append("drop decisions must not list supporting_fields")
    if invalid_paths:
        return None, invalid_paths
    return {
        "task_id": task_id,
        "decision": decision,
        "supporting_fields": supporting_fields,
        "reason": reason.strip(),
    }, []


def screen_tasks(
    blueprint: dict[str, Any], profile: dict[str, Any], overrides: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Screen each blueprint task separately and retry invalid field paths once."""
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Working blueprint must contain a tasks object.")
    valid_paths = set(profile.get("fields", {}))
    decisions: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            raise ValueError(f"Task {task_id} must be an object.")
        if overrides and task_id in overrides:
            decision, invalid = _normalize_decision(task_id, overrides[task_id], valid_paths)
            if decision is None:
                raise ValueError(f"Invalid user decision for {task_id}: {', '.join(invalid)}")
            decisions.append(decision)
            continue
        response = complete_json(_screening_prompt(task_id, task, profile), "Return the task support decision.")
        decision, invalid = _normalize_decision(task_id, response, valid_paths)
        if decision is None and invalid:
            retry_prompt = (
                _screening_prompt(task_id, task, profile)
                + f"\n\nYour previous response had invalid values: {invalid}. "
                + f"The complete legal annotation path list is: {sorted(valid_paths)}. Retry once."
            )
            response = complete_json(retry_prompt, "Return the corrected task support decision.")
            decision, invalid = _normalize_decision(task_id, response, valid_paths)
        if decision is None:
            raise TaskScreeningReviewRequired(task_id, response, sorted(valid_paths))
        decisions.append(decision)
    return decisions


def apply_task_screening(working_blueprint_path: str | Path, decisions: list[dict[str, Any]]) -> str:
    """Prune dropped tasks and write retained-task support metadata to the working blueprint."""
    blueprint_path = Path(working_blueprint_path)
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    tasks = blueprint["tasks"]
    decisions_by_task = {decision["task_id"]: decision for decision in decisions}
    for task_id in list(tasks):
        decision = decisions_by_task[task_id]
        if decision["decision"] == "drop":
            del tasks[task_id]
        else:
            tasks[task_id]["supporting_fields"] = decision["supporting_fields"]
            tasks[task_id]["dataset_support_reason"] = decision["reason"]
    blueprint_path.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = blueprint_path.with_name("task_screening_report.json")
    report_path.write_text(
        json.dumps({"decisions": decisions}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(report_path)
