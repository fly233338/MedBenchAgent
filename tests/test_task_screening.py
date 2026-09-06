import json
from pathlib import Path

import medbenchagent.task_screening as screening
import pytest
from medbenchagent.construction import event_context
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
    responses = iter([invalid, _plan()])
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda system, user: calls.append((system, user)) or next(responses),
    )

    plan = screening.generate_task_plan(_blueprint(), _profile())
    assert len(calls) == 2
    assert "complete Task Plan" in calls[0][0]
    assert "previous response violated this contract" in calls[1][0]
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


def test_complete_blueprint_task_plan_calls_model_once(monkeypatch) -> None:
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    task_ids = list(blueprint["tasks"])
    calls: list[tuple[str, str]] = []
    responses = {
        task_id: (
            {
                "decision": "retain",
                "supporting_fields": ["image.annotation.breast_birads"],
                "metric": "accuracy",
                "reason": "Available.",
            }
            if task_id == task_ids[0]
            else {"decision": "drop", "question_types": {}, "reason": "Unsupported."}
            if task_id in screening.DECOMPOSABLE_TASK_IDS
            else {
                "decision": "drop",
                "supporting_fields": [],
                "metric": None,
                "reason": "Unsupported.",
            }
        )
        for task_id in task_ids
    }

    def fake_complete_json(system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        return {"tasks": responses}

    monkeypatch.setattr(screening, "complete_json", fake_complete_json)

    result = screening.generate_task_plan(blueprint, _profile())

    assert list(result["tasks"]) == task_ids
    assert len(calls) == 1
    assert all(task_id in calls[0][0] for task_id in task_ids)
    assert "Blueprint task definitions" in calls[0][0]
    assert "Dataset profile" in calls[0][0]
    assert "image.annotation.breast_birads" in calls[0][0]
    assert result["tasks"][task_ids[0]]["metric"] == "accuracy"


def test_task_plan_requires_human_review_after_second_invalid_response(monkeypatch) -> None:
    invalid = {
        "tasks": {
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.not_present"],
                "metric": "macro_f1",
                "reason": "Invalid field.",
            },
            "diagnosis": _plan()["tasks"]["diagnosis"],
        }
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda system, user: calls.append((system, user)) or invalid,
    )

    with pytest.raises(screening.TaskPlanReviewRequired) as error:
        screening.generate_task_plan(_blueprint(), _profile())
    assert len(calls) == 2
    assert "unknown supporting fields" in error.value.errors[0]


def test_task_plan_revisions_only_replace_the_target_task(monkeypatch) -> None:
    original = _plan()
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda system, user: calls.append((system, user))
        or {
                "decision": "retain",
                "supporting_fields": ["image.annotation.findings[*].finding_type"],
                "metric": "accuracy",
                "reason": "Requested.",
            },
    )

    revised = screening.revise_task_plan(_blueprint(), _profile(), original, "diagnosis", "Keep this task.")

    assert revised["tasks"]["assessment_category"] == original["tasks"]["assessment_category"]
    assert revised["tasks"]["diagnosis"]["decision"] == "retain"
    assert len(calls) == 1
    assert "All Blueprint task definitions" in calls[0][0]
    assert "Complete current Task Plan" in calls[0][0]
    assert "assessment_category" in calls[0][0]
    assert "diagnosis" in calls[0][0]


def test_task_plan_revision_repairs_and_records_invalid_model_output(
    tmp_path: Path, monkeypatch
) -> None:
    invalid = {
        "decision": "retain",
        "supporting_fields": ["image.annotation.findings[*].finding_type"],
        "metric": "accuracy",
        "reason": "Requested.",
        "finding_field": "image.annotation.findings[*].finding_type",
    }
    valid = {key: value for key, value in invalid.items() if key != "finding_field"}
    responses = iter([invalid, valid])
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda system, user: calls.append((system, user)) or next(responses),
    )
    events_path = tmp_path / "construction_events.jsonl"

    with event_context(
        construction_events_path=str(events_path),
        run_id="run-1",
        agent="TaskPlanningAgent",
        operation="task_plan_revise",
    ):
        revised = screening.revise_task_plan(
            _blueprint(), _profile(), _plan(), "diagnosis", "Keep this task."
        )

    assert revised["tasks"]["diagnosis"] == valid
    assert len(calls) == 2
    assert "previous response violated this contract" in calls[1][0]
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events == [
        {
            "timestamp": events[0]["timestamp"],
            "event_type": "llm_validation_error",
            "run_id": "run-1",
            "task_id": "diagnosis",
            "agent": "TaskPlanningAgent",
            "operation": "task_plan_revise",
            "model": get_settings().llm.model,
            "validation_errors": [
                "diagnosis: decision must contain exactly decision, supporting_fields, metric, and reason"
            ],
            "model_output": invalid,
        }
    ]


def test_task_plan_revision_records_invalid_repair_output(tmp_path: Path, monkeypatch) -> None:
    responses = [
        {"decision": "retain", "attempt": "initial"},
        {"decision": "retain", "attempt": "repair"},
    ]
    remaining = iter(responses)
    monkeypatch.setattr(screening, "complete_json", lambda _system, _user: next(remaining))
    events_path = tmp_path / "construction_events.jsonl"

    with event_context(
        construction_events_path=str(events_path),
        run_id="run-1",
        agent="TaskPlanningAgent",
        operation="task_plan_revise",
    ):
        with pytest.raises(screening.TaskPlanReviewRequired):
            screening.revise_task_plan(
                _blueprint(), _profile(), _plan(), "diagnosis", "Keep this task."
            )

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["model_output"] for event in events] == responses
    assert [event["operation"] for event in events] == ["task_plan_revise", "repair"]


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


def test_full_plan_prompt_keeps_direct_axes_and_rejects_ordinal_relabeling(
    monkeypatch,
) -> None:
    blueprint = {
        "tasks": {
            "view_identification": {
                "description": "Identify the acquisition view.",
                "require": "Requires an acquisition-view field.",
                "response_format": "letter",
            },
            "anatomy_attribute": {
                "description": "Determine one global anatomical property.",
                "require": "Requires one property field per question type.",
                "response_format": "letter",
            },
            "assessment_category": {
                "description": "Determine the reporting category.",
                "require": "Requires a reporting-category field.",
                "response_format": "letter",
            },
            "severity_grading": {
                "description": "Determine severity.",
                "require": "Requires an explicit severity field.",
                "response_format": "letter",
            },
        }
    }
    profile = {
        "fields": {
            "image.annotation.view": {
                "distinct_value_count": 2,
                "top_values": ["front", "side"],
            },
            "image.annotation.texture": {
                "distinct_value_count": 2,
                "top_values": ["uniform", "heterogeneous"],
            },
            "image.annotation.reporting_category": {
                "distinct_value_count": 3,
                "top_values": ["A", "B", "C"],
            },
        }
    }
    response = {
        "tasks": {
            "view_identification": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.view"],
                "metric": "accuracy",
                "reason": "The acquisition view is explicit.",
            },
            "anatomy_attribute": {
                "decision": "retain",
                "question_types": {
                    "texture": {
                        "description": "Determine the global texture.",
                        "supporting_fields": ["image.annotation.texture"],
                        "metric": "accuracy",
                        "reason": "The texture is explicit.",
                    }
                },
                "reason": "One residual anatomical property is supported.",
            },
            "assessment_category": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.reporting_category"],
                "metric": "macro_f1",
                "reason": "The reporting category is explicit.",
            },
            "severity_grading": {
                "decision": "drop",
                "supporting_fields": [],
                "metric": None,
                "reason": "No field explicitly encodes severity.",
            },
        }
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        screening,
        "complete_json",
        lambda system, user: calls.append((system, user)) or response,
    )

    result = screening.generate_task_plan(blueprint, profile)

    assert len(calls) == 1
    assert "A decomposable attribute task is residual" in calls[0][0]
    assert "semantic question and answer axis" in calls[0][0]
    assert "A shared supporting field alone does not establish duplication" in calls[0][0]
    assert "ordered labels alone do not establish severity" in calls[0][0]
    assert "view_identification" in calls[0][0]
    assert "For example" not in calls[0][0]
    assert set(result["tasks"]["anatomy_attribute"]["question_types"]) == {"texture"}
    assert result["tasks"]["view_identification"]["decision"] == "retain"
    assert result["tasks"]["assessment_category"]["decision"] == "retain"
    assert result["tasks"]["severity_grading"]["decision"] == "drop"


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
    prompt = screening._task_plan_prompt(_blueprint()["tasks"], _profile())

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
    assert screening.VISUAL_GROUNDING_RULE in prompt
    assert screening.VISUAL_GROUNDING_PAIRING_RULE in prompt
    assert "Blueprint task definitions" in prompt
    assert "assessment_category" in prompt
    assert "diagnosis" in prompt
    assert '"description": "Diagnose"' in prompt


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
        {
            "detection": {"response_format": "bbox"},
        },
        {
            "tasks": {
                "detection": {
                    "decision": "drop",
                    "supporting_fields": [],
                    "metric": None,
                    "reason": "No bbox field.",
                }
            }
        },
        _profile(),
        "Retain detection using the mask.",
    )

    assert "A segmentation mask is not a bbox" in prompt


def test_visual_grounding_rule_is_shared_by_initial_and_revision_prompts() -> None:
    task = {"response_format": "bbox"}
    profile = {
        "fields": {
            "image.annotation.findings[*].bbox[*]": {},
            "image.annotation.findings[*].label": {},
        }
    }
    blueprint_tasks = {"visual_grounding": task}
    current_plan = {
        "tasks": {
            "visual_grounding": {
                "decision": "drop",
                "supporting_fields": [],
                "metric": None,
                "reason": "No.",
            }
        }
    }
    initial = screening._task_plan_prompt(blueprint_tasks, profile)
    revision = screening._task_revision_prompt(
        "visual_grounding",
        blueprint_tasks,
        current_plan,
        profile,
        "Review available fields.",
    )

    for prompt in (initial, revision):
        assert screening.VISUAL_GROUNDING_RULE in prompt
        assert screening.VISUAL_GROUNDING_PAIRING_RULE in prompt
        assert "BI-RADS" not in prompt
        assert "breast cancer" not in prompt.lower()
    assert "without a task ID or `tasks` wrapper" in revision
    assert "do not add finding_field or bbox_field keys" in revision
