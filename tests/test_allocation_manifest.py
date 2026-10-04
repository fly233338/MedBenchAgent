from medbenchagent.audit.manifest import build_manifest
from medbenchagent.planning.allocation import allocate_task_leaves


def _leaf(leaf_id, node_id="n", capacity=3):
    return {"leaf_task_id": leaf_id, "name": leaf_id, "scenario_id": node_id, "clinical_target_id": "finding", "question_templates": ["Q"], "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "target": "finding", "reference_rule": "read", "resolver": {"inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}], "steps": [{"op": "read", "selector": {"role": "image", "name": "image", "evidence_type": "asset"}}], "output_spec": "label", "result_step": "step_0"},
            "output_spec": "label", "task_paradigm": "VQA", "capability_category": "Recognition", "eligible_count": capacity}


def test_allocation_is_capacity_bounded_and_manifest_has_no_metric():
    leaves = [_leaf("a", capacity=1), _leaf("b", capacity=4)]
    allocation = allocate_task_leaves(leaves, 3)
    assert sum(allocation.values()) == 3
    assert allocation["a"] <= 1 and allocation["b"] <= 4
    manifest = build_manifest({"dataset_id": "d"}, [{"scenario_id": "n", "input_structure": "single-image", "operation": "recognize", "clinical_target": "finding", "capability_category": "Recognition", "validated_clinical_targets": [{"target_id": "finding", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["label"]}]}], leaves, allocation, item_count=3)
    assert manifest["schema_version"] == "benchmark-manifest"
    assert "metrics" not in manifest
    assert manifest["tasks"]["a"]["output_spec"] == "label"

