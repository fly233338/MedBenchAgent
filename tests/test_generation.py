from medbenchagent.generation.items import generate_task_items


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

