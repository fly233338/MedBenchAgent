import json
from pathlib import Path

import medbenchagent.task_screening as screening
import pytest
from medbenchagent.settings import get_settings


def _blueprint() -> dict:
    return {
        "tasks": {
            "assessment_category": {
                "description": "Assess category",
                "require": "BI-RADS",
                "task_type": "close_vqa",
                "response_format": "letter",
            },
            "diagnosis": {
                "description": "Diagnose",
                "require": "diagnosis",
                "task_type": "close_vqa",
                "response_format": "letter",
            },
        }
    }


def _profile() -> dict:
    return {
        "fields": {
            "image.annotation.breast_birads": {"distinct_value_count": 2},
            "image.annotation.findings[*].finding_type": {"distinct_value_count": 2},
        }
    }


def _plan() -> dict:
    return {
        "tasks": {
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.breast_birads"],
                "reason": "BI-RADS is available.",
            },
            "diagnosis": {"decision": "drop", "supporting_fields": [], "reason": "No diagnosis path."},
        }
    }


def test_task_plan_retries_one_invalid_complete_response_and_applies_only_after_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    invalid = {"tasks": {"assessment_category": _plan()["tasks"]["assessment_category"]}}
    responses = iter([invalid, _plan()])
    monkeypatch.setattr(screening, "complete_json", lambda *_args: next(responses))

    plan = screening.generate_task_plan(_blueprint(), _profile())
    blueprint_path = tmp_path / "blueprint.json"
    original = json.dumps(_blueprint(), sort_keys=True)
    blueprint_path.write_text(original, encoding="utf-8")
    assert blueprint_path.read_text(encoding="utf-8") == original

    report_path = Path(screening.apply_task_plan(blueprint_path, plan, _profile()))
    updated = json.loads(blueprint_path.read_text(encoding="utf-8"))
    assert list(updated["tasks"]) == ["assessment_category"]
    assert updated["tasks"]["assessment_category"]["supporting_fields"] == ["image.annotation.breast_birads"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == plan
    second_report_path = Path(screening.apply_task_plan(blueprint_path, plan, _profile()))
    assert json.loads(second_report_path.read_text(encoding="utf-8")) == plan


def test_complete_blueprint_task_plan_uses_one_model_call(monkeypatch) -> None:
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    task_ids = list(blueprint["tasks"])
    plan = {
        "tasks": {
            task_id: {"decision": "drop", "supporting_fields": [], "reason": "Unsupported."}
            for task_id in task_ids
        }
    }
    plan["tasks"][task_ids[0]] = {
        "decision": "retain",
        "supporting_fields": ["image.annotation.breast_birads"],
        "reason": "Available.",
    }
    calls = 0

    def fake_complete_json(*_args):
        nonlocal calls
        calls += 1
        return plan

    monkeypatch.setattr(screening, "complete_json", fake_complete_json)

    result = screening.generate_task_plan(blueprint, _profile())

    assert list(result["tasks"]) == task_ids
    assert calls == 1


def test_task_plan_requires_human_review_after_second_invalid_response(monkeypatch) -> None:
    invalid = {
        "tasks": {
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.not_present"],
                "reason": "Invalid field.",
            },
            "diagnosis": _plan()["tasks"]["diagnosis"],
        }
    }
    monkeypatch.setattr(screening, "complete_json", lambda *_args: invalid)

    with pytest.raises(screening.TaskPlanReviewRequired) as error:
        screening.generate_task_plan(_blueprint(), _profile())
    assert "unknown supporting fields" in error.value.errors[0]


def test_task_plan_revisions_only_replace_the_target_task(monkeypatch) -> None:
    original = _plan()
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda *_args: {"decision": "retain", "supporting_fields": ["image.annotation.findings[*].finding_type"], "reason": "Requested."},
    )

    revised = screening.revise_task_plan(_blueprint(), _profile(), original, "diagnosis", "Keep this task.")

    assert revised["tasks"]["assessment_category"] == original["tasks"]["assessment_category"]
    assert revised["tasks"]["diagnosis"]["decision"] == "retain"


def test_task_plan_rejects_unknown_tasks_and_all_dropped_tasks() -> None:
    unknown = {"tasks": {**_plan()["tasks"], "unknown": _plan()["tasks"]["diagnosis"]}}
    assert screening.validate_task_plan(_blueprint(), _profile(), unknown)[0] is None
    all_dropped = {"tasks": {task_id: {"decision": "drop", "supporting_fields": [], "reason": "No."} for task_id in _blueprint()["tasks"]}}
    _validated, errors = screening.validate_task_plan(_blueprint(), _profile(), all_dropped)
    assert errors == ["at least one task must be retained"]


def test_task_plan_rejects_constant_letter_support_but_not_non_letter_support() -> None:
    profile = {"fields": {"image.annotation.label": {"distinct_value_count": 1}}}
    retained = {
        "tasks": {
            "task": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.label"],
                "reason": "Available.",
            }
        }
    }

    letter, errors = screening.validate_task_plan(
        {"tasks": {"task": {"response_format": "letter"}}}, profile, retained
    )
    free_text, free_text_errors = screening.validate_task_plan(
        {"tasks": {"task": {"response_format": "free_text"}}}, profile, retained
    )

    assert letter is None
    assert errors == [
        "task: retained letter tasks need a supporting field with distinct_value_count >= 2"
    ]
    assert free_text_errors == []
    assert free_text == retained


def test_visual_grounding_and_detection_can_both_be_retained() -> None:
    blueprint = {
        "tasks": {
            "detection": {"response_format": "bbox"},
            "visual_grounding": {"response_format": "bbox"},
        }
    }
    profile = {"fields": {"image.annotation.findings[*].bbox[*]": {}, "image.annotation.findings[*].label": {}}}
    response = {
        "tasks": {
            "detection": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.findings[*].bbox[*]"],
                "reason": "Has a box.",
            },
            "visual_grounding": {
                "decision": "retain",
                "supporting_fields": [
                    "image.annotation.findings[*].bbox[*]",
                    "image.annotation.findings[*].label",
                ],
                "reason": "Has a paired label and box.",
            },
        }
    }

    plan, errors = screening.validate_task_plan(blueprint, profile, response)

    assert errors == []
    assert plan is not None
    assert plan["tasks"]["visual_grounding"]["decision"] == "retain"
    assert plan["tasks"]["detection"]["decision"] == "retain"
    assert plan["tasks"]["detection"]["supporting_fields"] == [
        "image.annotation.findings[*].bbox[*]"
    ]


def test_task_plan_prompt_checks_array_leaf_paths() -> None:
    prompt = screening._task_plan_prompt(_blueprint(), _profile())

    assert "For array-valued annotations" in prompt
    assert "`[*]` leaf path" in prompt
    assert "top_values" in prompt
    assert "distinct_value_count >= 2" in prompt
    assert "explicit positive and explicit negative" in prompt
    assert "may both be retained" in prompt
    assert "pathology-confirmed diagnoses are valid" in prompt
    assert "A segmentation mask is not a bbox" in prompt
    assert "derive a bbox from it" in prompt
    assert "description` and `require` fields" in prompt
    assert "bbox or mask" not in prompt
    assert "priority" not in prompt
    assert '"supporting_fields":["image.annotation.path"]' in prompt
    assert '"supporting_fields":["annotation.path"]' not in prompt


def test_task_revision_prompt_rejects_mask_as_bbox() -> None:
    prompt = screening._task_revision_prompt(
        "detection",
        {"response_format": "bbox"},
        {
            "decision": "drop",
            "supporting_fields": [],
            "reason": "No bbox field.",
        },
        _profile(),
        "Retain detection using the mask.",
    )

    assert "A segmentation mask is not a bbox" in prompt
