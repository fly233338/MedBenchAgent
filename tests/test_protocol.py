import copy
import json
from types import SimpleNamespace

import pytest

from medbenchagent.planning.protocol import (
    Protocol,
    resolve_answer,
    validate_dataset_protocol,
    template_specification,
)
from medbenchagent.generation.templates import create_templates, review_templates, render
from medbenchagent.generation.grounding import supports
from medbenchagent.settings import get_settings
from test_generation import _task, _record


def test_classification_counts_exactly_eligible_images():
    protocol = _task("letter", ["image.annotation.status"])["protocol"]
    rows = [
        _record(0, {"status": "benign"}),
        _record(1, {"status": "malignant"}),
        _record(2, {"status": "unknown"}),
        _record(3, {}),
    ]
    assert validate_dataset_protocol(protocol, rows) == 2
    assert [resolve_answer(r, protocol) is not None for r in rows] == [True, True, False, False]


def test_sample_scope_filters_capacity_and_answers():
    from medbenchagent.planning.protocol import resolve_eligibility
    protocol = _task("letter", ["image.annotation.status"])["protocol"]
    protocol["sample_conditions"] = [{"requirement_id": "us", "field": "image.modality",
                                      "operator": "in", "values": ["US"]}]
    rows = [_record(i, {"status": status}) for i, status in enumerate(["benign", "malignant", "benign"])]
    rows[0]["image"]["modality"] = "US"
    rows[1]["image"]["modality"] = "Mammography"
    rows[2]["image"].pop("modality", None)
    assert validate_dataset_protocol(protocol, rows) == 1
    assert [resolve_eligibility(r, protocol)["reason"] for r in rows] == ["eligible", "scope_mismatch", "scope_missing_field"]
    protocol["sample_conditions"][0]["operator"] = "not_in"
    assert [resolve_answer(r, protocol) is not None for r in rows] == [False, True, False]
    protocol["sample_conditions"][0]["field"] = "image.annotation.findings[*].status"
    with pytest.raises(ValueError, match="scalar field"):
        Protocol.model_validate(protocol)


def test_protocol_rejects_cross_entity_target_binding():
    task = _task(
        "bbox",
        ["image.annotation.findings[*].bbox[*]", "case.annotation.name"],
        target_field="case.annotation.name",
    )
    with pytest.raises(ValueError, match="same annotation entity"):
        Protocol.model_validate(task["protocol"])


def test_bound_target_and_bbox_pair_ignore_leaf_identity():
    fields = ["image.annotation.findings[*].bbox[*]", "image.annotation.findings[*].name"]
    task = _task("bbox", fields, target_field=fields[1])
    Protocol.model_validate(task["protocol"])
    row = _record(0, {"findings": [{"bbox": [1, 2, 3, 4], "name": "mass"}, {"name": "unrelated"}]})
    for identity in ("visual_grounding", "finding_attribute.margin", "opaque:123"):
        candidate = SimpleNamespace(
            record=row, task=task, task_id=identity, question_template="Locate {target}."
        )
        assert supports(candidate)
        assert render(candidate) == "Locate mass."
    row["image"]["annotation"]["findings"][0].pop("name")
    assert resolve_answer(row, task["protocol"]) is None


def test_explicit_text_binding_ignores_other_supporting_text():
    task = _task("free_text", ["image.annotation.report", "image.annotation.status"])
    row = _record(0, {"report": "Reference report", "status": "other text"})
    assert resolve_answer(row, task["protocol"])["answer"] == "Reference report"


def test_template_builder_and_reviewer_never_receive_answer_contract(tmp_path, monkeypatch):
    task = _task("letter", ["image.annotation.status"])
    task["protocol"]["answer_contract"]["value_mapping"][0]["raw_value"] = "PRIVATE_LABEL"
    task["evidence"] = [{"text": "PRIVATE_EVIDENCE"}]
    count = get_settings().generation.question_template_count
    prompts = []

    def model(system, prompt, *contracts, **kwargs):
        prompts.append(json.loads(prompt))
        if system.startswith("Review"):
            return {"verdict": "approved", "reason": "Valid specification"}
        return {"templates": [f"Question {i}: which category applies?" for i in range(count)]}

    monkeypatch.setattr("medbenchagent.generation.templates.complete_json", model)
    spec = template_specification(task["protocol"], "en")
    templates = create_templates(spec)
    review_templates(spec, templates)
    assert len(prompts) == 2
    for payload in prompts:
        text = json.dumps(payload)
        assert "PRIVATE" not in text
        assert "data_binding" not in text and "value_mapping" not in text
        assert payload["specification"]["language"] == "en"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(response_format="bbox"),
        lambda p: p.update(metric="miou"),
        lambda p: p.update(template_slots=["target"]),
        lambda p: p.update(generation_logic="legacy"),
    ],
)
def test_protocol_rejects_mismatched_or_legacy_fields(change):
    protocol = copy.deepcopy(_task("letter", ["image.annotation.status"])["protocol"])
    change(protocol)
    with pytest.raises(ValueError):
        Protocol.model_validate(protocol)


def test_template_interfaces_reject_full_task_before_model_call(monkeypatch):
    from medbenchagent.generation.templates import review_templates

    task = _task("letter", ["image.annotation.status"])
    monkeypatch.setattr(
        "medbenchagent.generation.templates.complete_json",
        lambda *a, **kw: pytest.fail("Full task crossed the template boundary"),
    )
    with pytest.raises(ValueError):
        create_templates(task)
    with pytest.raises(ValueError):
        review_templates(task, ["Which status?"])
    spec = template_specification(task["protocol"], "en").model_dump()
    spec["answer_contract"] = task["protocol"]["answer_contract"]
    with pytest.raises(ValueError):
        create_templates(spec)
