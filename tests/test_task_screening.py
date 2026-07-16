import json
from pathlib import Path

import medbenchagent.task_screening as screening
import pytest


def _blueprint() -> dict:
    return {
        "tasks": {
            "assessment_category": {"description": "Assess category", "require": "BI-RADS", "task_type": "close_vqa", "response_format": "letter"},
            "diagnosis": {"description": "Diagnose", "require": "diagnosis", "task_type": "close_vqa", "response_format": "letter"},
        }
    }


def _profile() -> dict:
    return {"fields": {"annotation.breast_birads": {}, "annotation.findings[*].finding_type": {}}}


def test_task_screening_retries_invalid_paths_and_prunes_blueprint(tmp_path: Path, monkeypatch) -> None:
    responses = iter(
        [
            {"task_id": "assessment_category", "decision": "retain", "supporting_fields": ["annotation.fake"], "reason": "bad"},
            {"task_id": "assessment_category", "decision": "retain", "supporting_fields": ["annotation.breast_birads"], "reason": "BI-RADS is available."},
            {"task_id": "diagnosis", "decision": "drop", "supporting_fields": [], "reason": "No diagnosis path."},
        ]
    )
    monkeypatch.setattr(screening, "complete_json", lambda *_args: next(responses))

    decisions = screening.screen_tasks(_blueprint(), _profile())
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(_blueprint()), encoding="utf-8")
    report_path = Path(screening.apply_task_screening(blueprint_path, decisions))
    updated = json.loads(blueprint_path.read_text(encoding="utf-8"))

    assert list(updated["tasks"]) == ["assessment_category"]
    assert updated["tasks"]["assessment_category"]["supporting_fields"] == ["annotation.breast_birads"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["decisions"] == decisions


def test_task_screening_requires_review_after_second_invalid_response(monkeypatch) -> None:
    invalid = {
        "task_id": "assessment_category",
        "decision": "retain",
        "supporting_fields": ["annotation.not_present"],
        "reason": "Invalid field.",
    }
    monkeypatch.setattr(screening, "complete_json", lambda *_args: invalid)

    with pytest.raises(screening.TaskScreeningReviewRequired) as error:
        screening.screen_tasks({"tasks": {"assessment_category": _blueprint()["tasks"]["assessment_category"]}}, _profile())
    assert error.value.valid_paths == ["annotation.breast_birads", "annotation.findings[*].finding_type"]
