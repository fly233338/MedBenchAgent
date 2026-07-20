import pytest

from medbenchagent.pdf_rag import validate_question_designs


def _plan(task_id: str, response_format: str, supporting_fields: list[str]) -> dict:
    return {
        "task_id": task_id,
        "task_type": "close_vqa" if response_format == "letter" else "open_vqa",
        "response_format": response_format,
        "decision": "retain",
        "supporting_fields": supporting_fields,
    }


def test_validate_question_designs_supports_letter_bbox_and_free_text() -> None:
    plans = [
        _plan("status", "letter", ["annotation.status"]),
        _plan("detection", "bbox", ["annotation.findings[*].kind", "annotation.findings[*].bbox[*]"]),
        _plan("report", "free_text", ["annotation.report"]),
    ]
    designs = [
        {
            "task_id": "status", "question_templates": ["What is the status?"],
            "placeholder_bindings": {}, "answer_list": ["benign", "malignant"],
            "answer_field": "annotation.status", "answer_value_mapping": {}, "validation_rules": [],
        },
        {
            "task_id": "detection", "question_templates": ["Locate the {target_text}."],
            "placeholder_bindings": {"target_text": "annotation.findings[*].kind"},
            "instance_path": "annotation.findings[*]", "target_text_field": "annotation.findings[*].kind",
            "bbox_field": "annotation.findings[*].bbox[*]", "bbox_format": "xyxy", "validation_rules": [],
        },
        {
            "task_id": "report", "question_templates": ["Generate a report."],
            "placeholder_bindings": {}, "answer_field": "annotation.report", "validation_rules": [],
        },
    ]

    result = validate_question_designs(plans, designs)

    assert [design["task_id"] for design in result] == ["status", "detection", "report"]
    assert result[1]["bbox_format"] == "xyxy"
    assert result[2]["answer_field"] == "annotation.report"


def test_validate_question_designs_rejects_fields_outside_task_support() -> None:
    plans = [_plan("report", "free_text", ["annotation.report"])]
    designs = [{
        "task_id": "report", "question_templates": ["Generate a report."],
        "placeholder_bindings": {}, "answer_field": "annotation.private_note", "validation_rules": [],
    }]

    with pytest.raises(ValueError, match="supporting field"):
        validate_question_designs(plans, designs)
