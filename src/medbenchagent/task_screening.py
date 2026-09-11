"""LLM-assisted task planning against a deterministic dataset profile."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from medbenchagent.llm import complete_json
from medbenchagent.construction import event_context, event_operation, record_llm_validation_error
from medbenchagent.run_artifacts import write_versioned_json
from medbenchagent.metrics import is_metric_supported, metric_prompt_catalog


class TaskPlanReviewRequired(ValueError):
    """The model twice returned a Task Plan response that violates the deterministic contract."""

    def __init__(self, response: dict[str, Any], errors: list[str]) -> None:
        self.response = response
        self.errors = errors
        super().__init__(f"Task Plan requires human review: {', '.join(errors)}")


DECOMPOSABLE_TASK_IDS = frozenset({"anatomy_attribute", "finding_attribute"})
_QUESTION_TYPE_ID = re.compile(r"[a-z][a-z0-9_]*")
VISUAL_GROUNDING_RULE = (
    "Visual grounding requires a bbox paired with one field that clearly identifies a single "
    "localized, image-observable finding or imaging sign; when several fields qualify, prefer "
    "the one whose values express the most visually direct and readily observable target."
)
VISUAL_GROUNDING_PAIRING_RULE = (
    "For array-valued findings, that field and bbox must belong to the same array item. A flat "
    "annotation record may pair that field with a unique bbox."
)


def _metric_prompt_rules(response_formats: set[str]) -> str:
    catalog = "\n".join(
        metric_prompt_catalog(response_format)
        for response_format in sorted(response_formats)
    )
    if not catalog:
        catalog = "No registered metric is compatible with the missing response format."
    return (
        "Metric selection rules:\n"
        "- A retained task must choose exactly one metric from the registered metrics compatible "
        "with its response format.\n"
        "- A dropped ordinary task must return `metric: null`; a dropped decomposable task must "
        "return an empty `question_types` object.\n"
        "- Metric selection evaluates model answers and must not be used as a reason to retain or drop "
        "a task.\n"
        "- Use the supplied task definition and dataset profile only. Do not invent class distributions "
        "or rely on unprovided external medical knowledge.\n\n"
        f"{catalog}\n"
    )


def _task_plan_prompt(
    blueprint_tasks: dict[str, Any], profile: dict[str, Any]
) -> str:
    response_formats = {
        task.get("response_format")
        for task in blueprint_tasks.values()
        if isinstance(task, dict) and isinstance(task.get("response_format"), str)
    }
    return (
        "Create one complete Task Plan for all predefined medical VLM benchmark tasks using only "
        "the supplied Blueprint definitions and dataset profile. Evaluate the tasks collectively "
        "before returning the plan.\n\n"
        'Return JSON only as `{"tasks": {...}}` and include every Blueprint task ID exactly once.\n'
        "Ordinary task entries use "
        '{"decision":"retain|drop","supporting_fields":["exact.path"],'
        '"metric":"registered_name_or_null","reason":"..."}.\n'
        "Decomposable task entries use "
        '{"decision":"retain|drop","question_types":{"lower_snake_case_id":'
        '{"description":"one observable property","supporting_fields":["exactly.one.path"],'
        '"metric":"registered_name","reason":"..."}},"reason":"..."}.\n'
        "A dropped decomposable task uses an empty `question_types` object. Keep every reason concise.\n\n"
        "Rules:\n"
        "1. Treat each task's `description` and `require` fields as the authoritative support criteria.\n"
        "2. `supporting_fields` must contain only exact paths present in `dataset_profile.fields`.\n"
        "3. Do not invent fields, annotation values, medical meanings, or relationships that are not "
        "supported by the dataset profile.\n"
        "4. `retain` requires supported evidence and a reason explaining how it satisfies the task.\n"
        "5. `drop` requires empty supporting fields or question types and a reason identifying the "
        "missing or insufficient evidence.\n"
        "6. Every retained letter-response task requires at least two semantically distinct values that "
        "can serve as correct answers. `distinct_value_count >= 2` is necessary but not sufficient: spelling "
        "variants, synonyms, or several raw values that map to one answer class still form one semantic class.\n\n"
        "Question-type rules:\n"
        "- A question type represents exactly one semantic answer axis.\n"
        "- Create one question type for each distinct observable property supported by the dataset.\n"
        "- Use exactly one supporting field for each question type.\n"
        "- Derive a concise lower_snake_case ID and description from the field path and values.\n"
        "- Do not assume a fixed catalogue of medical attributes.\n"
        "- Do not reuse one field in multiple question types under the same parent task.\n"
        "- Retain a question type only when at least two semantic values can be correct answers.\n"
        "- Retain the parent task only when it contains at least one valid question type.\n"
        "- Two fields belong to different question types when they answer different questions or cannot "
        "use one shared, mutually exclusive option taxonomy. Do not merge fields merely because they occur "
        "in the same annotation object or describe the same image or target.\n\n"
        "Whole-plan consistency:\n"
        "1. Treat each retained ordinary task and each retained question type as one atomic task.\n"
        "2. Re-check every atomic task against its parent Blueprint definition and the dataset profile.\n"
        "3. A decomposable attribute task is residual: omit any question type whose semantic answer axis "
        "is represented more directly by a retained ordinary task.\n"
        "4. Compare all atomic tasks by semantic question and answer axis. When two are duplicates, keep "
        "the one whose Blueprint definition most directly represents that axis.\n"
        "5. A shared supporting field alone does not establish duplication, and different metrics do not "
        "distinguish duplicate semantics.\n"
        "6. Remove unsupported or duplicate question types, and drop a decomposable parent when none remain.\n\n"
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
        "- `modality_identification` and `anatomy_identification`: require at least two semantically distinct "
        "annotated answer values. Drop either task when its answer field is constant.\n"
        "- `anatomy_attribute`: acquisition view, projection, plane, and orientation belong to "
        "`view_identification` and must not become anatomy question types.\n"
        "- `finding_presence` evaluates binary presence or absence when no direct categorical finding label is "
        "available. When a direct categorical finding label is available, prefer `finding_type` for the "
        "category-identification task. Require explicit positive and negative evidence; missing annotations, "
        "missing boxes, or missing masks do not establish absence.\n"
        "- `finding_type`: use only a field whose values directly name visual finding categories. Do not "
        "derive a finding type from an interpretation, diagnosis, assessment, pathology result, prognosis, "
        "or suspicion label. Drop the task when only indirect clinical fields are available.\n"
        "- `finding_description`: requires a paired expert-written natural-language description that can be "
        "used directly as the reference answer. Categorical labels, delimiter-joined terms, diagnoses, and "
        "isolated structured attributes are insufficient.\n"
        "- `diagnosis`: requires one clinically specific expert-confirmed diagnosis associated with the "
        "finding or case. Imaging or pathology-confirmed diagnoses are valid. Finding types, assessment "
        "categories, benign/malignant status, and missing or non-diagnostic values are insufficient.\n"
        "- `pathology_status`: requires an explicit pathology or malignancy-status field. A diagnosis field "
        "alone is insufficient.\n"
        "- `severity_grading`: ordered labels alone do not establish severity. The field must explicitly "
        "encode severity, extent, grade, or stage rather than another clinical concept.\n"
        f"- {VISUAL_GROUNDING_RULE} {VISUAL_GROUNDING_PAIRING_RULE}\n"
        "- `detection`: requires candidate bbox support. Other finding or case annotations may coexist, but "
        "they must not be used to identify the target in a detection question.\n\n"
        "Additional rules:\n"
        "- `visual_grounding` and `detection` are independent capabilities and may both be retained when "
        "their respective requirements are satisfied.\n"
        "- If there is no candidate bbox support, drop both spatial tasks.\n"
        "- For tasks without additional criteria above, apply their Blueprint `description` and `require` "
        "fields directly.\n\n"
        f"{_metric_prompt_rules(response_formats)}"
        f"Blueprint task definitions:\n{json.dumps(blueprint_tasks, ensure_ascii=False)}\n\n"
        f"Dataset profile:\n{json.dumps(profile, ensure_ascii=False)}"
    )


def _task_revision_prompt(
    task_id: str,
    blueprint_tasks: dict[str, Any],
    current_task_plan: dict[str, Any],
    profile: dict[str, Any],
    feedback: str,
) -> str:
    task = blueprint_tasks[task_id]
    decomposable = task_id in DECOMPOSABLE_TASK_IDS
    response_format = task.get("response_format")
    schema = (
        "Return decision, question_types, and reason. A retained task needs at least one question type; "
        "a dropped task needs an empty question_types object. Each question type needs a lower_snake_case "
        "ID, description, exactly one supporting field, metric, and reason."
        if decomposable
        else "Return exactly this top-level JSON shape, without a task ID or `tasks` wrapper and without "
        "additional fields: "
        '{"decision":"retain|drop","supporting_fields":["exact.path"],'
        '"metric":"registered_name_or_null","reason":"..."}. '
        "A retained task needs at least one supporting field; a dropped task needs an empty "
        "supporting_fields array and metric null. For visual_grounding, put both the paired finding "
        "and bbox paths in supporting_fields; do not add finding_field or bbox_field keys."
    )
    return (
        "Revise only the selected Task Plan entry using the user's feedback. Use the complete "
        "Blueprint and current Task Plan as consistency context, but do not return or change any "
        "other entry. Return JSON only.\n\n"
        f"{schema}\n\nSupporting "
        "fields must exist in dataset_profile.fields. Use top_values as the primary field-semantic evidence; "
        "they may be truncated and are not exhaustive. Every retained letter-response task requires at least "
        "two semantically distinct values that can be correct answers; distinct_value_count >= 2 is necessary "
        "but spelling variants, synonyms, or raw values mapping to one answer class do not qualify.\n\n"
        "finding_presence evaluates binary presence or absence when no direct categorical finding label is "
        "available. When a direct categorical finding label is available, prefer finding_type for the "
        "category-identification task. Require explicit positive and negative evidence; missing "
        "annotations, missing boxes, or missing masks do not establish absence.\n\nfinding_description requires "
        "paired expert-written "
        "natural-language text, not categorical labels, delimiter-joined terms, diagnoses, or isolated "
        "attributes.\n\ndiagnosis requires one clinically specific imaging or pathology-confirmed diagnosis, "
        "not a finding type, assessment category, benign/malignant status, or missing value.\n\n"
        "finding_type requires values that directly name visual finding categories; interpretations, "
        "diagnoses, assessments, pathology results, prognoses, and suspicion labels are insufficient.\n\n"
        "For anatomy_attribute and finding_attribute, one question type represents one semantic answer axis; "
        "different properties require separate question types, each with one unique supporting field.\n\n"
        "A decomposable attribute task is residual: omit any question type whose semantic answer axis is "
        "represented more directly by a retained ordinary task. A shared field alone does not establish "
        "duplication, and different metrics do not distinguish duplicate semantics.\n\n"
        "For anatomy_attribute, acquisition view, projection, plane, and orientation belong to "
        "view_identification. For severity_grading, ordered labels qualify only when the field explicitly "
        "encodes severity, extent, grade, or stage.\n\n"
        f"{VISUAL_GROUNDING_RULE} {VISUAL_GROUNDING_PAIRING_RULE}\n\n"
        "detection requires a bbox and remains supported even when finding text coexists. Both spatial tasks "
        "may be retained. A segmentation mask is not a bbox and must not be treated or converted into one.\n\n"
        f"{_metric_prompt_rules({response_format} if isinstance(response_format, str) else set())}"
        f"\nSelected Task ID: {task_id}\n"
        f"Selected Blueprint task: {json.dumps(task, ensure_ascii=False)}\n\n"
        f"All Blueprint task definitions: {json.dumps(blueprint_tasks, ensure_ascii=False)}\n\n"
        f"Complete current Task Plan: {json.dumps(current_task_plan, ensure_ascii=False)}\n\n"
        f"User feedback: {feedback}\n\n"
        f"Dataset profile: {json.dumps(profile, ensure_ascii=False)}"
    )


def _field_has_multiple_values(path: str, profile: dict[str, Any]) -> bool:
    field = profile.get("fields", {}).get(path)
    count = field.get("distinct_value_count") if isinstance(field, dict) else None
    return isinstance(count, int) and not isinstance(count, bool) and count >= 2


def _normalize_question_types(
    task_id: str,
    response: object,
    valid_paths: set[str],
    profile: dict[str, Any],
    response_format: object,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(response, dict) or set(response) != {"decision", "question_types", "reason"}:
        return None, [f"{task_id}: decomposable decision must contain decision, question_types, and reason"]
    decision, question_types, reason = response["decision"], response["question_types"], response["reason"]
    errors: list[str] = []
    if decision not in {"retain", "drop"}:
        errors.append(f"{task_id}: decision must be retain or drop")
    if not isinstance(reason, str) or not reason.strip():
        errors.append(f"{task_id}: reason is required")
    if not isinstance(question_types, dict):
        errors.append(f"{task_id}: question_types must be an object")
        return None, errors
    if decision == "retain" and not question_types:
        errors.append(f"{task_id}: retain needs at least one question type")
    if decision == "drop" and question_types:
        errors.append(f"{task_id}: drop must use an empty question_types object")
    normalized: dict[str, dict[str, Any]] = {}
    used_fields: set[str] = set()
    for question_type_id, question_type in question_types.items():
        prefix = f"{task_id}.{question_type_id}"
        if not isinstance(question_type_id, str) or not _QUESTION_TYPE_ID.fullmatch(question_type_id):
            errors.append(f"{prefix}: question type ID must be lower_snake_case")
            continue
        if not isinstance(question_type, dict) or set(question_type) != {
            "description", "supporting_fields", "metric", "reason"
        }:
            errors.append(
                f"{prefix}: question type must contain description, supporting_fields, metric, and reason"
            )
            continue
        description = question_type["description"]
        fields = question_type["supporting_fields"]
        metric = question_type["metric"]
        type_reason = question_type["reason"]
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{prefix}: description is required")
        if not isinstance(type_reason, str) or not type_reason.strip():
            errors.append(f"{prefix}: reason is required")
        if not is_metric_supported(response_format, metric):
            errors.append(f"{prefix}: unsupported metric for response format: {metric}")
        if not isinstance(fields, list) or len(fields) != 1 or not isinstance(fields[0], str):
            errors.append(f"{prefix}: exactly one supporting field is required")
            continue
        field = fields[0]
        if field not in valid_paths:
            errors.append(f"{prefix}: unknown supporting field: {field}")
        if field in used_fields:
            errors.append(f"{task_id}: supporting field is reused by multiple question types: {field}")
        used_fields.add(field)
        if field in valid_paths and not _field_has_multiple_values(field, profile):
            errors.append(f"{prefix}: supporting field needs distinct_value_count >= 2")
        normalized[question_type_id] = {
            "description": description.strip() if isinstance(description, str) else "",
            "supporting_fields": [field],
            "metric": metric,
            "reason": type_reason.strip() if isinstance(type_reason, str) else "",
        }
    if errors:
        return None, errors
    return {
        "decision": decision,
        "question_types": normalized,
        "reason": reason.strip(),
    }, []


def _normalize_task_decision(
    task_id: str,
    response: object,
    valid_paths: set[str],
    profile: dict[str, Any],
    response_format: object = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if task_id in DECOMPOSABLE_TASK_IDS:
        return _normalize_question_types(
            task_id, response, valid_paths, profile, response_format
        )
    if not isinstance(response, dict):
        return None, [f"{task_id}: decision must be an object"]
    expected_keys = {"decision", "supporting_fields", "metric", "reason"}
    if set(response) != expected_keys:
        return None, [
            f"{task_id}: decision must contain exactly decision, supporting_fields, metric, and reason"
        ]
    decision = response.get("decision")
    supporting_fields = response.get("supporting_fields")
    metric = response.get("metric")
    reason = response.get("reason")
    if decision not in {"retain", "drop"}:
        return None, [f"{task_id}: decision must be retain or drop"]
    if not isinstance(supporting_fields, list) or not all(isinstance(field, str) for field in supporting_fields):
        return None, [f"{task_id}: supporting_fields must be a list of strings"]
    if not isinstance(reason, str) or not reason.strip():
        return None, [f"{task_id}: reason is required"]
    if decision == "retain" and not is_metric_supported(response_format, metric):
        return None, [f"{task_id}: unsupported metric for response format: {metric}"]
    if decision == "drop" and metric is not None:
        return None, [f"{task_id}: dropped task must use metric null"]
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
        "metric": metric,
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
        decision, decision_errors = _normalize_task_decision(
            task_id,
            planned_tasks[task_id],
            valid_paths,
            profile,
            tasks[task_id].get("response_format") if isinstance(tasks[task_id], dict) else None,
        )
        errors.extend(decision_errors)
        if decision is not None:
            normalized_tasks[task_id] = decision
            task = tasks[task_id]
            if (
                task_id not in DECOMPOSABLE_TASK_IDS
                and task.get("response_format") == "letter"
                and decision["decision"] == "retain"
            ):
                has_multiple_values = any(
                    isinstance(field_profile := profile.get("fields", {}).get(field), dict)
                    and isinstance(count := field_profile.get("distinct_value_count"), int)
                    and not isinstance(count, bool)
                    and count >= 2
                    for field in decision["supporting_fields"]
                )
                if not has_multiple_values:
                    errors.append(
                        f"{task_id}: retained letter tasks need a supporting field with "
                        "distinct_value_count >= 2"
                    )
    if not errors and not any(plan["decision"] == "retain" for plan in normalized_tasks.values()):
        errors.append("at least one task must be retained")
    if errors:
        return None, errors
    return {"tasks": {task_id: normalized_tasks[task_id] for task_id in tasks}}, []


def generate_task_plan(blueprint: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Generate and validate one complete Task Plan, repairing one invalid response."""
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Working blueprint must contain a tasks object")
    prompt = _task_plan_prompt(tasks, profile)
    response = complete_json(prompt, "Return the complete Task Plan.")
    plan, errors = validate_task_plan(blueprint, profile, response)
    if plan is not None:
        return plan
    with event_operation("repair"):
        repaired = complete_json(
            f"{prompt}\n\nYour previous response violated this contract: "
            f"{json.dumps(errors, ensure_ascii=False)}. Retry once with a complete valid Task Plan.",
            "Return the corrected complete Task Plan.",
        )
    plan, errors = validate_task_plan(blueprint, profile, repaired)
    if plan is None:
        raise TaskPlanReviewRequired(
            repaired if isinstance(repaired, dict) else {"response": repaired},
            errors,
        )
    return plan


def revise_task_plan(
    blueprint: dict[str, Any], profile: dict[str, Any], draft_task_plan: dict[str, Any], task_id: str, feedback: str
) -> dict[str, Any]:
    """Return a replacement for exactly one draft Task Plan entry."""
    tasks = blueprint.get("tasks")
    planned_tasks = draft_task_plan.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(planned_tasks, dict) or task_id not in tasks or task_id not in planned_tasks:
        raise ValueError(f"Unknown task_id: {task_id}")
    prompt = _task_revision_prompt(task_id, tasks, draft_task_plan, profile, feedback)

    def validate_revision(response: object) -> tuple[dict[str, Any] | None, list[str]]:
        decision, errors = _normalize_task_decision(
            task_id,
            response,
            set(profile.get("fields", {})),
            profile,
            tasks[task_id].get("response_format"),
        )
        if decision is None:
            return None, errors
        merged = {"tasks": {**planned_tasks, task_id: decision}}
        return validate_task_plan(blueprint, profile, merged)

    with event_context(task_id=task_id):
        response = complete_json(prompt, "Return the revised task plan entry.")
        validated, errors = validate_revision(response)
        if validated is not None:
            return validated

        record_llm_validation_error(errors=errors, model_output=response)
        with event_operation("repair"):
            repaired = complete_json(
                f"{prompt}\n\nYour previous response violated this contract: "
                f"{json.dumps(errors, ensure_ascii=False)}. Previous response: "
                f"{json.dumps(response, ensure_ascii=False)}. Retry once with only the corrected "
                "selected task entry.",
                "Return the corrected revised task plan entry.",
            )
            validated, errors = validate_revision(repaired)
            if validated is None:
                record_llm_validation_error(errors=errors, model_output=repaired)
                raise TaskPlanReviewRequired(repaired, errors)
        return validated


def write_task_plan_report(
    working_blueprint_path: str | Path, task_plan: dict[str, Any]
) -> dict[str, str | int]:
    """Persist the current Task Plan as a new immutable run artifact version."""
    path = Path(working_blueprint_path)
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    blueprint["task_plan"] = task_plan
    return write_versioned_json(path.parent, "blueprints", blueprint)


def apply_task_plan(
    working_blueprint_path: str | Path,
    task_plan: dict[str, Any],
    profile: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> str:
    """Apply a confirmed Task Plan once and preserve the complete plan in its report."""
    blueprint_path = Path(working_blueprint_path)
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if not isinstance(task_plan.get("tasks"), dict):
        raise ValueError("Cannot apply invalid Task Plan: tasks must be an object")
    current_tasks = blueprint.get("tasks")
    if not isinstance(current_tasks, dict):
        raise ValueError("Working blueprint must contain a tasks object")
    validation_profile = profile or {"fields": {}}
    planned_task_ids = set(task_plan["tasks"])
    current_task_ids = set(current_tasks)
    current_parent_ids = {
        task.get("parent_task_id", task_id) if isinstance(task, dict) else task_id
        for task_id, task in current_tasks.items()
    }
    already_expanded = current_parent_ids <= planned_task_ids and current_parent_ids != current_task_ids
    if current_task_ids == planned_task_ids:
        validation_blueprint = blueprint
    elif already_expanded or current_task_ids < planned_task_ids and all(
        task_plan["tasks"][task_id].get("decision") == "drop" for task_id in planned_task_ids - current_task_ids
    ):
        validation_tasks: dict[str, dict[str, Any]] = {}
        for task_id in task_plan["tasks"]:
            current = current_tasks.get(task_id)
            if not isinstance(current, dict) and already_expanded:
                current = next(
                    (
                        candidate
                        for candidate in current_tasks.values()
                        if isinstance(candidate, dict)
                        and candidate.get("parent_task_id") == task_id
                    ),
                    None,
                )
            validation_tasks[task_id] = current if isinstance(current, dict) else {}
        validation_blueprint = {"tasks": validation_tasks}
    else:
        raise ValueError("Working Blueprint tasks do not match the confirmed Task Plan")
    validated, errors = validate_task_plan(validation_blueprint, validation_profile, task_plan)
    if validated is None:
        raise ValueError(f"Cannot apply invalid Task Plan: {', '.join(errors)}")
    tasks = current_tasks
    if not already_expanded:
        for task_id in list(tasks):
            decision = validated["tasks"][task_id]
            if decision["decision"] == "drop":
                del tasks[task_id]
            elif task_id in DECOMPOSABLE_TASK_IDS:
                parent = tasks.pop(task_id)
                for question_type_id, question_type in decision["question_types"].items():
                    expanded = deepcopy(parent)
                    expanded["parent_task_id"] = task_id
                    expanded["question_type_id"] = question_type_id
                    expanded["description"] = question_type["description"]
                    expanded["supporting_fields"] = question_type["supporting_fields"]
                    expanded["dataset_support_reason"] = question_type["reason"]
                    expanded["metric"] = question_type["metric"]
                    tasks[f"{task_id}.{question_type_id}"] = expanded
            else:
                tasks[task_id]["supporting_fields"] = decision["supporting_fields"]
                tasks[task_id]["dataset_support_reason"] = decision["reason"]
                tasks[task_id]["metric"] = decision["metric"]
    else:
        for task_id, decision in validated["tasks"].items():
            if decision["decision"] != "retain" or task_id not in DECOMPOSABLE_TASK_IDS:
                continue
            for question_type_id, question_type in decision["question_types"].items():
                expanded = tasks.get(f"{task_id}.{question_type_id}")
                if isinstance(expanded, dict):
                    expanded["metric"] = question_type["metric"]
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
        f"{task_id}.{question_type_id}" if task_id in DECOMPOSABLE_TASK_IDS else task_id
        for task_id, decision in validated["tasks"].items()
        if decision["decision"] == "retain"
        for question_type_id in (
            decision["question_types"] if task_id in DECOMPOSABLE_TASK_IDS else [""]
        )
    }
    written = json.loads(blueprint_path.read_text(encoding="utf-8"))
    if set(written.get("tasks", {})) != expected_retained:
        raise ValueError("Working Blueprint does not match the confirmed retained tasks")
    if report_path is not None:
        return str(report_path)
    return str(write_task_plan_report(blueprint_path, validated)["path"])
