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
                "metric": "macro_f1",
                "reason": "BI-RADS is available.",
            },
            "diagnosis": {
                "decision": "drop",
                "supporting_fields": [],
                "metric": None,
                "reason": "No diagnosis path.",
            },
        }
    }


def test_task_plan_retries_one_invalid_complete_response_and_applies_only_after_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    invalid = {
        "decision": "retain",
        "supporting_fields": ["image.annotation.breast_birads"],
        "reason": "Missing metric.",
    }
    responses = iter(
        [
            invalid,
            _plan()["tasks"]["assessment_category"],
            _plan()["tasks"]["diagnosis"],
            _plan(),
        ]
    )
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
    assert updated["tasks"]["assessment_category"]["metric"] == "macro_f1"
    assert json.loads(report_path.read_text(encoding="utf-8")) == plan
    second_report_path = Path(screening.apply_task_plan(blueprint_path, plan, _profile()))
    assert json.loads(second_report_path.read_text(encoding="utf-8")) == plan


def test_complete_blueprint_task_plan_calls_model_once_per_task(monkeypatch) -> None:
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    task_ids = list(blueprint["tasks"])
    calls: list[tuple[str, str]] = []
    task_responses: dict[str, dict] = {}

    def fake_complete_json(system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        task_id = next(
            (task_id for task_id in task_ids if f"Task ID: {task_id}" in system_prompt), None
        )
        if task_id is None:
            return {"tasks": task_responses}
        if task_id == task_ids[0]:
            response = {
                "decision": "retain",
                "supporting_fields": ["image.annotation.breast_birads"],
                "metric": "accuracy",
                "reason": "Available.",
            }
        elif task_id in screening.DECOMPOSABLE_TASK_IDS:
            response = {"decision": "drop", "question_types": {}, "reason": "Unsupported."}
        else:
            response = {
                "decision": "drop",
                "supporting_fields": [],
                "metric": None,
                "reason": "Unsupported.",
            }
        task_responses[task_id] = response
        return response

    monkeypatch.setattr(screening, "complete_json", fake_complete_json)

    result = screening.generate_task_plan(blueprint, _profile())

    assert list(result["tasks"]) == task_ids
    assert len(calls) == len(task_ids) + 1
    assert all(
        sum(f"Task ID: {task_id}" in system_prompt for task_id in task_ids) == 1
        for system_prompt, _user_prompt in calls[:-1]
    )
    assert "Review the complete draft Task Plan" in calls[-1][0]
    assert all(task_id in calls[-1][0] for task_id in task_ids)
    assert result["tasks"][task_ids[0]]["metric"] == "accuracy"


def test_task_plan_requires_human_review_after_second_invalid_response(monkeypatch) -> None:
    invalid = {
        "decision": "retain",
        "supporting_fields": ["image.annotation.not_present"],
        "metric": "macro_f1",
        "reason": "Invalid field.",
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
        lambda *_args: {
            "decision": "retain",
            "supporting_fields": ["image.annotation.findings[*].finding_type"],
            "metric": "accuracy",
            "reason": "Requested.",
        }
    )

    revised = screening.revise_task_plan(_blueprint(), _profile(), original, "diagnosis", "Keep this task.")

    assert revised["tasks"]["assessment_category"] == original["tasks"]["assessment_category"]
    assert revised["tasks"]["diagnosis"]["decision"] == "retain"


def test_task_plan_rejects_unknown_tasks_and_all_dropped_tasks() -> None:
    unknown = {"tasks": {**_plan()["tasks"], "unknown": _plan()["tasks"]["diagnosis"]}}
    assert screening.validate_task_plan(_blueprint(), _profile(), unknown)[0] is None
    all_dropped = {
        "tasks": {
            task_id: (
                {"decision": "drop", "question_types": {}, "reason": "No."}
                if task_id in screening.DECOMPOSABLE_TASK_IDS
                else {"decision": "drop", "supporting_fields": [], "metric": None, "reason": "No."}
            )
            for task_id in _blueprint()["tasks"]
        }
    }
    _validated, errors = screening.validate_task_plan(_blueprint(), _profile(), all_dropped)
    assert errors == ["at least one task must be retained"]


def test_task_plan_rejects_constant_letter_support_but_not_non_letter_support() -> None:
    profile = {"fields": {"image.annotation.label": {"distinct_value_count": 1}}}
    retained = {
        "tasks": {
            "task": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.label"],
                "metric": "accuracy",
                "reason": "Available.",
            }
        }
    }

    letter, errors = screening.validate_task_plan(
        {"tasks": {"task": {"response_format": "letter"}}}, profile, retained
    )
    free_text_response = {
        "tasks": {
            "task": {
                **retained["tasks"]["task"],
                "metric": "bleu",
            }
        }
    }
    free_text, free_text_errors = screening.validate_task_plan(
        {"tasks": {"task": {"response_format": "free_text"}}},
        profile,
        free_text_response,
    )

    assert letter is None
    assert errors == [
        "task: retained letter tasks need a supporting field with distinct_value_count >= 2"
    ]
    assert free_text_errors == []
    assert free_text == free_text_response


def test_task_plan_rejects_incompatible_metric() -> None:
    response = {
        "tasks": {
            "task": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.label"],
                "metric": "bleu",
                "reason": "Available.",
            }
        }
    }
    plan, errors = screening.validate_task_plan(
        {"tasks": {"task": {"response_format": "letter"}}},
        {"fields": {"image.annotation.label": {"distinct_value_count": 2}}},
        response,
    )

    assert plan is None
    assert errors == ["task: unsupported metric for response format: bleu"]


def test_reconciliation_drops_clear_duplicate_and_rewrites_reason(monkeypatch) -> None:
    blueprint = {
        "tasks": {
            "assessment_category": {
                "description": "Determine the standardized assessment category.",
                "require": "Requires a standardized assessment field.",
                "response_format": "letter",
            },
            "severity_grading": {
                "description": "Determine the severity or grade.",
                "require": "Requires an ordered severity or grade field.",
                "response_format": "letter",
            },
        }
    }
    profile = {
        "fields": {
            "image.annotation.birads": {
                "distinct_value_count": 7,
                "top_values": ["1", "2", "3"],
            }
        }
    }
    draft = {
        "tasks": {
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.birads"],
                "metric": "macro_f1",
                "reason": "The field is available.",
            },
            "severity_grading": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.birads"],
                "metric": "macro_f1",
                "reason": "The field is ordered.",
            },
        }
    }
    reviewed = {
        "tasks": {
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.birads"],
                "metric": "macro_f1",
                "reason": "The field directly supplies the assessment categories.",
            },
            "severity_grading": {
                "decision": "drop",
                "supporting_fields": [],
                "metric": None,
                "reason": "This task duplicates the retained assessment-category task.",
            },
        }
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda system, user: calls.append((system, user)) or reviewed,
    )

    result = screening._reconcile_task_plan(blueprint, profile, draft)

    assert result == reviewed
    assert len(calls) == 1
    assert "Review the complete draft Task Plan" in calls[0][0]
    assert "assessment_category" in calls[0][0]
    assert "severity_grading" in calls[0][0]
    assert "ordered category values alone do not make them distinct" in calls[0][0]
    assert "Draft Task Plan" in calls[0][0]


def test_reconciliation_cannot_change_supporting_fields_or_metric(monkeypatch) -> None:
    draft = _plan()
    invalid = {
        "tasks": {
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.findings[*].finding_type"],
                "metric": "accuracy",
                "reason": "Changed by review.",
            },
            "diagnosis": draft["tasks"]["diagnosis"],
        }
    }
    monkeypatch.setattr(screening, "complete_json", lambda *_args: invalid)

    with pytest.raises(screening.TaskPlanReviewRequired) as error:
        screening._reconcile_task_plan(_blueprint(), _profile(), draft)

    assert any("cannot change supporting_fields" in item for item in error.value.errors)
    assert any("cannot change metric" in item for item in error.value.errors)


def test_reconciliation_can_remove_one_question_type_without_adding_another(monkeypatch) -> None:
    blueprint = {
        "tasks": {
            "finding_attribute": {
                "response_format": "letter",
            }
        }
    }
    profile = {
        "fields": {
            "image.annotation.shape": {"distinct_value_count": 3},
            "image.annotation.signal_intensity": {"distinct_value_count": 2},
        }
    }
    draft = {
        "tasks": {
            "finding_attribute": {
                "decision": "retain",
                "question_types": {
                    "shape": {
                        "description": "Determine shape.",
                        "supporting_fields": ["image.annotation.shape"],
                        "metric": "accuracy",
                        "reason": "Shape is available.",
                    },
                    "signal_intensity": {
                        "description": "Determine signal intensity.",
                        "supporting_fields": ["image.annotation.signal_intensity"],
                        "metric": "macro_f1",
                        "reason": "Signal intensity is available.",
                    },
                },
                "reason": "Both properties are available.",
            }
        }
    }
    reviewed = {
        "tasks": {
            "finding_attribute": {
                "decision": "retain",
                "question_types": {
                    "shape": {
                        "description": "Determine shape.",
                        "supporting_fields": ["image.annotation.shape"],
                        "metric": "accuracy",
                        "reason": "Shape remains supported.",
                    }
                },
                "reason": "The signal-intensity question was removed as a duplicate.",
            }
        }
    }
    monkeypatch.setattr(screening, "complete_json", lambda *_args: reviewed)
    result = screening._reconcile_task_plan(blueprint, profile, draft)

    assert set(result["tasks"]["finding_attribute"]["question_types"]) == {"shape"}


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
                "metric": "miou",
                "reason": "Has a box.",
            },
            "visual_grounding": {
                "decision": "retain",
                "supporting_fields": [
                    "image.annotation.findings[*].bbox[*]",
                    "image.annotation.findings[*].label",
                ],
                "metric": "miou",
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
    prompt = screening._task_plan_prompt(
        "assessment_category", _blueprint()["tasks"]["assessment_category"], _profile()
    )

    assert "For array-valued annotations" in prompt
    assert "`[*]` leaf path" in prompt
    assert "top_values" in prompt
    assert "distinct_value_count >= 2" in prompt
    assert "binary presence or absence" in prompt
    assert "direct categorical finding label" in prompt
    assert "prefer `finding_type`" in prompt
    assert "explicit positive and negative" in prompt
    assert "may both be retained" in prompt
    assert "pathology-confirmed diagnoses are valid" in prompt
    assert "A segmentation mask is not a bbox" in prompt
    assert "derive a bbox from it" in prompt
    assert "description` and `require` fields" in prompt
    assert "bbox or mask" not in prompt
    assert "priority" not in prompt
    assert '"supporting_fields":["exact.path"]' in prompt
    assert '"supporting_fields":["annotation.path"]' not in prompt
    assert "Question-type rules" in prompt
    assert "interpretation, diagnosis, assessment" in prompt
    assert "Metric selection rules" in prompt
    assert "macro_f1" in prompt
    assert "Task ID: assessment_category" in prompt
    assert "Task ID: diagnosis" not in prompt
    assert '"description": "Diagnose"' not in prompt


def test_attribute_task_plan_expands_question_types_into_independent_tasks(tmp_path: Path) -> None:
    blueprint = {
        "tasks": {
            "finding_attribute": {
                "description": "Determine one finding property.",
                "require": "One structured property.",
                "response_format": "letter",
                "category": "finding_recognition_and_characterization",
                "sample_count": None,
            }
        }
    }
    profile = {
        "fields": {
            "image.annotation.shape": {"distinct_value_count": 3},
            "image.annotation.signal_intensity": {"distinct_value_count": 2},
        }
    }
    plan = {
        "tasks": {
            "finding_attribute": {
                "decision": "retain",
                "question_types": {
                    "shape": {
                        "description": "Determine the finding shape.",
                        "supporting_fields": ["image.annotation.shape"],
                        "metric": "accuracy",
                        "reason": "Three shape values are available.",
                    },
                    "signal_intensity": {
                        "description": "Determine the signal intensity.",
                        "supporting_fields": ["image.annotation.signal_intensity"],
                        "metric": "macro_f1",
                        "reason": "Two intensity values are available.",
                    },
                },
                "reason": "Two independent properties are supported.",
            }
        }
    }
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps(blueprint), encoding="utf-8")

    report = screening.apply_task_plan(path, plan, profile)
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]

    assert list(tasks) == ["finding_attribute.shape", "finding_attribute.signal_intensity"]
    assert tasks["finding_attribute.shape"]["parent_task_id"] == "finding_attribute"
    assert tasks["finding_attribute.shape"]["question_type_id"] == "shape"
    assert tasks["finding_attribute.shape"]["metric"] == "accuracy"
    assert tasks["finding_attribute.signal_intensity"]["metric"] == "macro_f1"
    assert json.loads(Path(report).read_text(encoding="utf-8")) == plan


def test_attribute_task_plan_rejects_reused_or_constant_fields() -> None:
    blueprint = {"tasks": {"anatomy_attribute": {"response_format": "letter"}}}
    profile = {
        "fields": {
            "study.annotation.size": {"distinct_value_count": 1},
        }
    }
    response = {
        "tasks": {
            "anatomy_attribute": {
                "decision": "retain",
                "question_types": {
                    "size": {
                        "description": "Determine size.",
                        "supporting_fields": ["study.annotation.size"],
                        "metric": "accuracy",
                        "reason": "Available.",
                    },
                    "size_group": {
                        "description": "Determine size group.",
                        "supporting_fields": ["study.annotation.size"],
                        "metric": "accuracy",
                        "reason": "Available.",
                    },
                },
                "reason": "Available.",
            }
        }
    }

    plan, errors = screening.validate_task_plan(blueprint, profile, response)

    assert plan is None
    assert any("reused by multiple question types" in error for error in errors)
    assert any("distinct_value_count >= 2" in error for error in errors)


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
