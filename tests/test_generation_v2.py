from medbenchagent.generation.v2 import generate_task_items


def test_v2_generation_uses_resolver_and_hides_reference_from_inputs():
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

