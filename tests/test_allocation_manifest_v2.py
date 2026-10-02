from medbenchagent.audit.manifest_v2 import build_manifest
from medbenchagent.planning.allocation_v2 import allocate_task_leaves


def _leaf(leaf_id, node_id="n", capacity=3):
    return {"leaf_id": leaf_id, "name": leaf_id, "node_id": node_id, "question": "Q", "input": [{"role": "image"}],
            "target": "finding", "answer": "read", "resolver": {"inputs": [{"role": "image"}], "steps": [{"op": "read", "selector": {"role": "image"}}], "output_spec": "label", "result_step": "step_0"},
            "output_spec": "label", "task_paradigm": "VQA", "capability_category": "Recognition", "eligible_count": capacity}


def test_allocation_is_capacity_bounded_and_manifest_has_no_metric():
    leaves = [_leaf("a", capacity=1), _leaf("b", capacity=4)]
    allocation = allocate_task_leaves(leaves, 3)
    assert sum(allocation.values()) == 3
    assert allocation["a"] <= 1 and allocation["b"] <= 4
    manifest = build_manifest({"dataset_id": "d"}, [{"node_id": "n", "input_structure": "single-image", "operation": "recognize", "clinical_target": "finding", "capability_category": "Recognition"}], leaves, allocation, item_count=3)
    assert manifest["schema_version"] == "benchmark-manifest.v2"
    assert "metrics" not in manifest
    assert manifest["tasks"]["a"]["output_spec"] == "label"
