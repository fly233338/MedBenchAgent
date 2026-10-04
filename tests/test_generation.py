from medbenchagent.generation.items import generate_task_items
from medbenchagent.audit.task_items import audit_task_items
from pydantic import ValidationError


def test_generation_uses_resolver_and_hides_reference_from_inputs():
    leaf = {
        "leaf_id": "report-leaf", "name": "report", "node_id": "node-1",
        "question": "Describe the study.", "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "report", "answer": "Read the paired report", "output_spec": "text",
        "task_paradigm": "Report Generation", "capability_category": "Generation",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [
                {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
            ], "output_spec": "text", "result_step": "step_1",
        },
    }
    case = {"case_id": "case-1", "evidence": [
        {"id": "image-ev", "type": "asset", "name": "image", "value": "asset-1", "ref": []},
        {"id": "report-ev", "type": "text", "name": "report", "value": "Clear.", "ref": ["image-ev"]},
    ]}
    items = generate_task_items([case], [leaf], "dataset")
    assert len(items) == 1
    assert items[0]["answer"] == "Clear."
    assert items[0]["input_assets"] == ["asset-1"]
    assert items[0]["input_evidence_ids"] == ["image-ev"]
    assert items[0]["reference_evidence_ids"] == ["report-ev"]


def test_generation_honors_allocation_and_case_order():
    leaf = {
        "leaf_id": "l", "name": "label", "node_id": "n", "question": "What is shown?",
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "answer": "Read label", "output_spec": "label",
        "task_paradigm": "VQA", "capability_category": "Recognition",
        "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                     "steps": [{"op": "read", "selector": {"role": "label", "name": "label", "evidence_type": "value"}}],
                     "output_spec": "label", "result_step": "step_0"},
    }
    cases = [{"case_id": "b", "evidence": [{"id": "i", "type": "asset", "name": "image", "value": "b"},
                                                {"id": "l", "type": "value", "name": "label", "value": "yes"}]},
             {"case_id": "a", "evidence": [{"id": "i", "type": "asset", "name": "image", "value": "a"},
                                                {"id": "l", "type": "value", "name": "label", "value": "yes"}]}]
    items = generate_task_items(cases, [leaf], "d", allocation={"l": 1})
    assert len(items) == 1 and items[0]["case_id"] == "a"


def test_generation_uses_reproducible_balanced_template_cycle():
    leaf = {
        "leaf_id": "l", "name": "label", "node_id": "n",
        "question_templates": ["What is shown?", "Which finding is present?", "Identify the finding."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "answer": "Read label", "output_spec": "label",
        "task_paradigm": "VQA", "capability_category": "Recognition",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "label", "name": "label", "evidence_type": "value"}}],
            "output_spec": "label", "result_step": "step_0",
        },
    }
    cases = [
        {"case_id": str(index), "evidence": [
            {"id": "i", "type": "asset", "name": "image", "value": str(index)},
            {"id": "l", "type": "value", "name": "label", "value": "yes"},
        ]}
        for index in range(6)
    ]

    first = generate_task_items(cases, [leaf], "d", seed=11)
    second = generate_task_items(cases, [leaf], "d", seed=11)

    assert [(item["case_id"], item["template_index"], item["question"]) for item in first] == [
        (item["case_id"], item["template_index"], item["question"]) for item in second
    ]
    assert {item["template_index"] for item in first} == {0, 1, 2}
    assert all(item["template_count"] == 3 for item in first)


def test_generation_preserves_input_role_order_and_derived_label_options():
    leaf = {
        "leaf_id": "compare-leaf", "name": "compare", "node_id": "scenario-compare",
        "question_templates": ["Is Image A larger than Image B?"],
        "options": ["true", "false"],
        "input": [
            {"role": "image_a", "name": "image_a", "evidence_type": "asset"},
            {"role": "image_b", "name": "image_b", "evidence_type": "asset"},
        ],
        "target": "measurement", "reference_rule": "Compare the two measurements.",
        "output_spec": "label", "task_paradigm": "VQA", "capability_category": "Comparison",
        "resolver": {
            "inputs": [
                {"role": "image_a", "name": "image_a", "evidence_type": "asset"},
                {"role": "image_b", "name": "image_b", "evidence_type": "asset"},
            ],
            "steps": [
                {"op": "read", "selector": {"role": "image_a", "name": "measurement_a", "evidence_type": "value"}},
                {"op": "read", "selector": {"role": "image_b", "name": "measurement_b", "evidence_type": "value"}},
                {"op": "compare", "left": "step_0", "right": "step_1", "mode": "greater"},
            ],
            "output_spec": "label", "result_step": "step_2",
        },
    }
    case = {"case_id": "c", "evidence": [
        {"id": "b-image", "type": "asset", "name": "image_b", "value": "asset-b", "ref": []},
        {"id": "b-value", "type": "value", "name": "measurement_b", "value": 2, "ref": ["b-image"]},
        {"id": "a-image", "type": "asset", "name": "image_a", "value": "asset-a", "ref": []},
        {"id": "a-value", "type": "value", "name": "measurement_a", "value": 3, "ref": ["a-image"]},
    ]}

    items = generate_task_items([case], [leaf], "dataset")

    assert items[0]["answer"] == "true"
    assert items[0]["options"] == ["true", "false"]
    assert items[0]["input_evidence_ids"] == ["a-image", "b-image"]
    assert items[0]["input_assets"] == ["asset-a", "asset-b"]
    assert items[0]["input_bindings"] == [
        {"role": "image_a", "evidence_ids": ["a-image"], "assets": ["asset-a"]},
        {"role": "image_b", "evidence_ids": ["b-image"], "assets": ["asset-b"]},
    ]


def test_generation_and_audit_use_map_label_output_domain():
    leaf = {
        "leaf_id": "margin-parent", "name": "margin", "node_id": "scenario-margin",
        "question_templates": ["How is the margin classified?"],
        "options": ["circumscribed", "not circumscribed"],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "attribute", "reference_rule": "Map documented margin subtypes to their supported parent class.",
        "output_spec": "label", "task_paradigm": "VQA", "capability_category": "Recognition",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [
                {"op": "read", "selector": {"role": "margin", "name": "margin", "evidence_type": "value"}},
                {"op": "map_label", "input": "step_0", "mapping": {
                    "circumscribed": "circumscribed",
                    "not circumscribed - angular": "not circumscribed",
                }},
            ],
            "output_spec": "label", "result_step": "step_1",
        },
    }
    case = {"case_id": "c", "evidence": [
        {"id": "image", "type": "asset", "name": "image", "value": "asset", "ref": []},
        {"id": "margin", "type": "value", "name": "margin", "value": "not circumscribed - angular", "ref": []},
    ]}
    items = generate_task_items([case], [leaf], "dataset")
    assert items[0]["answer"] == "not circumscribed"
    assert audit_task_items([case], items, [leaf])["accepted"] == 1


def test_generation_rejects_duplicate_asset_binding_across_roles():
    leaf = {
        "leaf_id": "duplicate-assets", "name": "compare", "node_id": "scenario-compare",
        "question_templates": ["Compare A and B."],
        "input": [
            {"role": "image_a", "name": "image", "evidence_type": "asset"},
            {"role": "image_b", "name": "image", "evidence_type": "asset"},
        ],
        "target": "measurement", "reference_rule": "Compare.",
        "output_spec": "number", "task_paradigm": "VQA", "capability_category": "Comparison",
        "resolver": {
            "inputs": [
                {"role": "image_a", "name": "image", "evidence_type": "asset"},
                {"role": "image_b", "name": "image", "evidence_type": "asset"},
            ],
            "steps": [
                {"op": "read", "selector": {"role": "image_a", "name": "value_a", "evidence_type": "value"}},
                {"op": "read", "selector": {"role": "image_b", "name": "value_b", "evidence_type": "value"}},
                {"op": "compare", "left": "step_0", "right": "step_1", "mode": "difference"},
            ],
            "output_spec": "number", "result_step": "step_2",
        },
    }
    case = {"case_id": "c", "evidence": [
        {"id": "image", "type": "asset", "name": "image", "value": "asset", "ref": []},
        {"id": "a", "type": "value", "name": "value_a", "value": 3, "ref": ["image"]},
        {"id": "b", "type": "value", "name": "value_b", "value": 2, "ref": ["image"]},
    ]}

    assert generate_task_items([case], [leaf], "dataset") == []


def test_leaf_rejects_unresolved_question_placeholders():
    leaf = {
        "leaf_id": "placeholder", "name": "label", "node_id": "scenario",
        "question_templates": ["Is {{image_a}} larger than {{image_b}}?"],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "Read label", "output_spec": "label",
        "task_paradigm": "VQA", "capability_category": "Recognition",
        "resolver": {
            "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "label", "name": "label", "evidence_type": "value"}}],
            "output_spec": "label", "result_step": "step_0",
        },
    }

    try:
        from medbenchagent.planning.task_space import LeafTask
        LeafTask.model_validate(leaf)
    except ValidationError as exc:
        assert "unresolved placeholders" in str(exc)
    else:
        raise AssertionError("unresolved question placeholders were accepted")

