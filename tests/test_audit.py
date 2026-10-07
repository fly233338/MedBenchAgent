from image_fixtures import image_context
from medbenchagent.planning.resolver import resolve_task_input
from medbenchagent.planning.task_space import LeafTask
from medbenchagent.audit.task_items import audit_task_items


def test_audit_reconstructs_reference_and_rejects_leakage(tmp_path):
    leaf = {"leaf_task_id": "l", "name": "report", "scenario_id": "n", "clinical_target_id": "finding", "question_templates": ["Describe."],
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}], "target": "finding",
            "reference_rule": "Read report", "output_spec": "text", "task_paradigm": "Report Generation",
            "capability_category": "Generation", "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                      {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}}],
            "output_spec": "text", "result_step": "step_1"}}
    case = {"case_id": "c", "evidence": [{"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []},
                                              {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]}]}
    good = {
        "task_id": "l", "case_id": "c", "answer": "Clear",
        "question": "Describe.", "template_index": 0, "template_count": 1,
        "options": None,
        "input_bindings": [{"role": "image", "evidence_ids": ["a"], "assets": ["asset"]}],
        "input_assets": ["asset"],
        "input_evidence_ids": ["a"], "reference_evidence_ids": ["r"],
    }
    bad = {**good, "id": "bad", "input_evidence_ids": ["a", "r"]}
    context = image_context(tmp_path, [case])
    good["input_presentations"] = resolve_task_input(case, LeafTask.model_validate(leaf), None, context).presentations
    report = audit_task_items([case], [good, bad], [leaf], resolver_context=context)
    assert report["accepted"] == 1 and report["rejected"] == 1
    assert report["reason_counts"]["input_reference_leakage"] == 1


def test_audit_rejects_wrong_question_and_input_binding(tmp_path):
    leaf = {
        "leaf_task_id": "l", "name": "report", "scenario_id": "n", "clinical_target_id": "finding", "question_templates": ["Describe."],
        "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
        "target": "finding", "reference_rule": "Read report", "output_spec": "text",
        "task_paradigm": "Report Generation", "capability_category": "Generation",
        "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                     "steps": [
                         {"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}},
                         {"op": "reference_text", "selector": {"role": "reference", "name": "report", "evidence_type": "text"}},
                     ], "output_spec": "text", "result_step": "step_1"},
    }
    case = {"case_id": "c", "evidence": [
        {"id": "a", "type": "asset", "name": "image", "value": "asset", "ref": []},
        {"id": "r", "type": "text", "name": "report", "value": "Clear", "ref": ["a"]},
    ]}
    item = {
        "task_id": "l", "case_id": "c", "answer": "Clear", "question": "Wrong.",
        "template_index": 0, "template_count": 1, "options": None,
        "input_bindings": [{"role": "image", "evidence_ids": ["r"], "assets": []}],
        "input_assets": [], "input_evidence_ids": ["r"], "reference_evidence_ids": ["r"],
    }

    report = audit_task_items([case], [item], [leaf], resolver_context=image_context(tmp_path, [case]))

    assert report["accepted"] == 0
    assert {"question_mismatch", "input_binding_mismatch", "input_order_mismatch"} <= set(report["reason_counts"])

