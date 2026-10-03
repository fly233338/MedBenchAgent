from medbenchagent.audit.task_items import audit_task_items


def test_audit_reconstructs_reference_and_rejects_leakage():
    leaf = {"leaf_id": "l", "name": "report", "node_id": "n", "question": "Describe.",
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}], "target": "finding",
            "answer": "Read report", "output_spec": "text", "task_paradigm": "Report Generation",
            "capability_category": "Generation", "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                      {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}],
            "output_spec": "text", "result_step": "step_1"}}
    case = {"case_id": "c", "evidence": [{"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []},
                                              {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]}]}
    good = {"task_id": "l", "case_id": "c", "answer": "Clear", "input_evidence_ids": ["a"], "reference_evidence_ids": ["r"]}
    bad = {**good, "id": "bad", "input_evidence_ids": ["a", "r"]}
    report = audit_task_items([case], [good, bad], [leaf])
    assert report["accepted"] == 1 and report["rejected"] == 1
    assert report["reason_counts"]["input_reference_leakage"] == 1

