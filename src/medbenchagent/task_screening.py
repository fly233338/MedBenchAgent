"""LLM-assisted task planning against a deterministic dataset profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medbenchagent.llm import complete_json
from medbenchagent.construction import event_operation


class TaskPlanReviewRequired(ValueError):
    """The model twice returned a Task Plan that violates the deterministic contract."""

    def __init__(self, response: dict[str, Any], errors: list[str]) -> None:
        self.response = response
        self.errors = errors
        super().__init__(f"Task Plan requires human review: {', '.join(errors)}")


def _task_plan_prompt(
    blueprint: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    return (
        "Determine which predefined medical VLM benchmark tasks are supported by the dataset's "
        "entity-scoped annotation fields. Use only the supplied Blueprint tasks and dataset profile.\n\n"

        "Return JSON only, using exactly this schema:\n"
        '{"tasks":{"task_id":{"decision":"retain|drop",'
        '"supporting_fields":["image.annotation.path"],"reason":"..."}}}\n\n'

        "Rules:\n"
        "1. `tasks` must contain every task in the Blueprint, and no others.\n"
        "2. Treat each task's `description` and `require` fields as the authoritative support criteria.\n"
        "3. `supporting_fields` must contain only exact paths present in `dataset_profile.fields`.\n"
        "4. Do not invent fields, annotation values, medical meanings, or relationships that are not "
        "supported by the dataset profile.\n"
        "5. `retain` requires at least one supporting field and a reason explaining how the supplied "
        "evidence satisfies the task requirements.\n"
        "6. `drop` requires an empty `supporting_fields` list and a reason identifying the missing or "
        "insufficient evidence.\n"
        "7. At least one task must be retained.\n\n"

        "How to interpret the dataset profile:\n"
        "- For semantic fields, use `top_values` as the primary evidence when present. These are bounded, "
        "possibly truncated examples and are not the field's complete value space.\n"
        "- Use `types`, `non_empty_images`, and `distinct_value_count` as supporting evidence about field "
        "availability and value structure. They do not establish medical meaning by themselves.\n"
        "- Use `dataset_profile.structures` to understand object and array nesting.\n"
        "- For array-valued annotations, inspect the `[*]` leaf paths before rejecting the parent field. "
        "A parent path may be null or absent from `dataset_profile.fields` while its leaf paths contain "
        "usable values.\n"
        "- When two fields must be paired within an array item, their paths must share the same relevant "
        "`[*]` parent.\n\n"

        "Spatial annotation rules:\n"
        "- Candidate bbox support requires an explicit bbox leaf path, normally ending in `bbox[*]`, "
        "with numeric values and at least one non-empty image.\n"
        "- A missing `top_values` entry for a bbox field does not make the field unsupported.\n"
        "- A segmentation mask is not a bbox. Do not treat a mask as a bbox, derive a bbox from it, "
        "or assume that downstream code will convert it into one.\n"
        "- If the profile contains only mask annotations and no separate bbox field, drop both "
        "`visual_grounding` and `detection`.\n"
        "- Exact bbox coordinate count, xyxy validity, and per-image uniqueness are validated "
        "deterministically downstream.\n\n"

        "Task-specific criteria:\n"
        "- `modality_identification` and `anatomy_identification`: one valid annotated value is sufficient.\n"
        "- `finding_presence`: requires explicit positive or negative evidence. Missing finding, bbox, mask, "
        "or other annotation values do not establish absence.\n"
        "- `finding_description`: requires a paired expert-written natural-language description that can be "
        "used directly as the reference answer. Categorical labels, delimiter-joined terms, diagnoses, and "
        "isolated structured attributes are insufficient.\n"
        "- `diagnosis`: requires one clinically specific expert-confirmed diagnosis associated with the "
        "finding or case. Imaging or pathology-confirmed diagnoses are valid. Finding types, assessment "
        "categories, benign/malignant status, and missing or non-diagnostic values are insufficient.\n"
        "- `pathology_status`: requires an explicit pathology or malignancy-status field. A diagnosis field "
        "alone is insufficient.\n"
        "- `visual_grounding`: requires candidate bbox support plus a finding name, structured label, or "
        "description paired with that box. For array-valued findings, the identifying field and bbox must "
        "belong to the same array item. A flat annotation record may pair a finding field with a unique bbox.\n"
        "- `detection`: requires candidate bbox support. Other finding or case annotations may coexist, but "
        "they must not be used to identify the target in a detection question.\n\n"

        "Additional rules:\n"
        "- `visual_grounding` and `detection` are independent capabilities and may both be retained when "
        "their respective requirements are satisfied.\n"
        "- If there is no candidate bbox support, drop both spatial tasks.\n"
        "- For tasks without additional criteria above, apply their Blueprint `description` and `require` "
        "fields directly.\n\n"

        f"Blueprint tasks:\n{json.dumps(blueprint['tasks'], ensure_ascii=False)}\n\n"
        f"Dataset profile:\n{json.dumps(profile, ensure_ascii=False)}"
    )


def _task_revision_prompt(
    task_id: str,
    task: dict[str, Any],
    current_plan: dict[str, Any],
    profile: dict[str, Any],
    feedback: str,
) -> str:
    return (
        "Revise only this one Task Plan entry using the user's feedback. Return JSON only with "
        "decision, supporting_fields, and reason. decision must be retain or drop. A retain decision "
        "needs at least one supporting field. A drop decision needs no supporting fields. Supporting "
        "fields must exist in dataset_profile.fields. Use top_values as the primary field-semantic evidence; "
        "they may be truncated and are not exhaustive. finding_description requires paired expert-written "
        "natural-language text, not categorical labels, delimiter-joined terms, diagnoses, or isolated "
        "attributes. diagnosis requires one clinically specific imaging or pathology-confirmed diagnosis, "
        "not a finding type, assessment category, benign/malignant status, or missing value. "
        "visual_grounding requires a bbox paired with a finding name, structured label, or description. "
        "detection requires a bbox and remains supported even when finding text coexists. Both spatial tasks "
        "may be retained. A segmentation mask is not a bbox and must not be treated or converted into one.\n\n"
        f"Task ID: {task_id}\nTask: {json.dumps(task, ensure_ascii=False)}\n"
        f"Current task plan: {json.dumps(current_plan, ensure_ascii=False)}\n"
        f"User feedback: {feedback}\n\n"
        f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}"
    )


def _normalize_task_decision(
    task_id: str, response: object, valid_paths: set[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(response, dict):
        return None, [f"{task_id}: decision must be an object"]
    decision = response.get("decision")
    supporting_fields = response.get("supporting_fields")
    reason = response.get("reason")
    if decision not in {"retain", "drop"}:
        return None, [f"{task_id}: decision must be retain or drop"]
    if not isinstance(supporting_fields, list) or not all(isinstance(field, str) for field in supporting_fields):
        return None, [f"{task_id}: supporting_fields must be a list of strings"]
    if not isinstance(reason, str) or not reason.strip():
        return None, [f"{task_id}: reason is required"]
    invalid_paths = sorted(set(supporting_fields) - valid_paths)
    if invalid_paths:
        return None, [f"{task_id}: unknown supporting fields: {', '.join(invalid_paths)}"]
    if decision == "retain" and not supporting_fields:
        return None, [f"{task_id}: retain needs at least one supporting field"]
    if decision == "drop" and supporting_fields:
        return None, [f"{task_id}: drop must not list supporting fields"]
    return {
        "decision": decision,
        "supporting_fields": supporting_fields,
        "reason": reason.strip(),
    }, []


def validate_task_plan(blueprint: dict[str, Any], profile: dict[str, Any], response: object) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a complete Task Plan without relying on model judgement."""
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict):
        return None, ["Working blueprint must contain a tasks object"]
    if not isinstance(response, dict) or set(response) != {"tasks"} or not isinstance(response["tasks"], dict):
        return None, ["Task Plan must be an object containing only a tasks object"]
    planned_tasks = response["tasks"]
    expected_task_ids = set(tasks)
    actual_task_ids = set(planned_tasks)
    errors: list[str] = []
    if missing := sorted(expected_task_ids - actual_task_ids):
        errors.append(f"missing tasks: {', '.join(missing)}")
    if unknown := sorted(actual_task_ids - expected_task_ids):
        errors.append(f"unknown tasks: {', '.join(unknown)}")
    valid_paths = set(profile.get("fields", {}))
    normalized_tasks: dict[str, dict[str, Any]] = {}
    for task_id in sorted(expected_task_ids & actual_task_ids):
        decision, decision_errors = _normalize_task_decision(task_id, planned_tasks[task_id], valid_paths)
        errors.extend(decision_errors)
        if decision is not None:
            normalized_tasks[task_id] = decision
    if not errors and not any(plan["decision"] == "retain" for plan in normalized_tasks.values()):
        errors.append("at least one task must be retained")
    if errors:
        return None, errors
    return {"tasks": {task_id: normalized_tasks[task_id] for task_id in tasks}}, []


def generate_task_plan(blueprint: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Generate all task decisions in one model call and repair one invalid response."""
    prompt = _task_plan_prompt(blueprint, profile)
    response = complete_json(prompt, "Return the complete Task Plan.")
    plan, errors = validate_task_plan(blueprint, profile, response)
    if plan is not None:
        return plan
    with event_operation("repair"):
        repaired = complete_json(
            f"{prompt}\n\nYour previous response violated this contract: {errors}. Retry once with a valid complete Task Plan.",
            "Return the corrected complete Task Plan.",
        )
    plan, errors = validate_task_plan(blueprint, profile, repaired)
    if plan is None:
        raise TaskPlanReviewRequired(repaired, errors)
    return plan


def revise_task_plan(
    blueprint: dict[str, Any], profile: dict[str, Any], draft_task_plan: dict[str, Any], task_id: str, feedback: str
) -> dict[str, Any]:
    """Return a replacement for exactly one draft Task Plan entry."""
    tasks = blueprint.get("tasks")
    planned_tasks = draft_task_plan.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(planned_tasks, dict) or task_id not in tasks or task_id not in planned_tasks:
        raise ValueError(f"Unknown task_id: {task_id}")
    response = complete_json(
        _task_revision_prompt(task_id, tasks[task_id], planned_tasks[task_id], profile, feedback),
        "Return the revised task plan entry.",
    )
    decision, errors = _normalize_task_decision(task_id, response, set(profile.get("fields", {})))
    if decision is None:
        raise TaskPlanReviewRequired(response, errors)
    merged = {"tasks": {**planned_tasks, task_id: decision}}
    validated, errors = validate_task_plan(blueprint, profile, merged)
    if validated is None:
        raise TaskPlanReviewRequired(merged, errors)
    return validated


def task_plan_entries(task_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the stable Task Plan contract to downstream list-based consumers."""
    return [{"task_id": task_id, **decision} for task_id, decision in task_plan["tasks"].items()]


def apply_task_plan(
    working_blueprint_path: str | Path, task_plan: dict[str, Any], profile: dict[str, Any] | None = None
) -> str:
    """Apply a confirmed Task Plan once and preserve the complete plan in its report."""
    blueprint_path = Path(working_blueprint_path)
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if not isinstance(task_plan.get("tasks"), dict):
        raise ValueError("Cannot apply invalid Task Plan: tasks must be an object")
    current_tasks = blueprint.get("tasks")
    if not isinstance(current_tasks, dict):
        raise ValueError("Working blueprint must contain a tasks object")
    validation_profile = profile or {
        "fields": {field: {} for plan in task_plan["tasks"].values() for field in plan["supporting_fields"]}
    }
    planned_task_ids = set(task_plan["tasks"])
    current_task_ids = set(current_tasks)
    if current_task_ids == planned_task_ids:
        validation_blueprint = blueprint
    elif current_task_ids < planned_task_ids and all(
        task_plan["tasks"][task_id].get("decision") == "drop" for task_id in planned_task_ids - current_task_ids
    ):
        validation_blueprint = {"tasks": {task_id: {} for task_id in task_plan["tasks"]}}
    else:
        raise ValueError("Working Blueprint tasks do not match the confirmed Task Plan")
    validated, errors = validate_task_plan(validation_blueprint, validation_profile, task_plan)
    if validated is None:
        raise ValueError(f"Cannot apply invalid Task Plan: {', '.join(errors)}")
    tasks = current_tasks
    for task_id in list(tasks):
        decision = validated["tasks"][task_id]
        if decision["decision"] == "drop":
            del tasks[task_id]
        else:
            tasks[task_id]["supporting_fields"] = decision["supporting_fields"]
            tasks[task_id]["dataset_support_reason"] = decision["reason"]
    temporary = blueprint_path.with_suffix(blueprint_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(blueprint_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    expected_retained = {
        task_id for task_id, decision in validated["tasks"].items() if decision["decision"] == "retain"
    }
    written = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if set(written.get("tasks", {})) != expected_retained:
        raise ValueError("Working Blueprint does not match the confirmed retained tasks")
    report_path = blueprint_path.with_name("task_plan_report.json")
    report_path.write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(report_path)
