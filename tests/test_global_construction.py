from image_fixtures import image_context
import json

from medbenchagent.agents.finalization import balance_option_positions
from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.models import ImageEntry, CaseRecord, Evidence
from medbenchagent.generation.items import generate_task_items, write_task_items
from medbenchagent.planning.allocation import allocate_task_plan
from medbenchagent.planning.task_space import (
    CapabilityCategory,
    LeafTask,
    TaskPlanState,
    TaskScenario,
)


def _plan(dataset_id, leaves):
    scenario_id = f"scenario-{dataset_id}"
    parsed = {
        leaf_id: LeafTask.model_validate({
            "leaf_task_id": leaf_id,
            "name": leaf_id,
            "scenario_id": scenario_id,
            "clinical_target_id": "finding",
            "question_templates": ["What is shown?"],
            "input": [{"role": "image", "name": "image", "evidence_type": "asset"}],
            "target": "finding",
            "reference_rule": "Read the label.",
            "resolver": {
                "inputs": [{"role": "image", "name": "image", "evidence_type": "asset"}],
                "steps": [{"op": "read", "selector": {"role": "label", "name": "label", "evidence_type": "value"}}],
                "output_spec": "label",
                "result_step": "step_0",
            },
            "output_spec": "label",
            "task_paradigm": "VQA",
            "capability_category": "Recognition",
            "options": ["yes", "no"],
            "eligible_count": capacity,
        })
        for leaf_id, capacity in leaves.items()
    }
    scenario = TaskScenario(
        scenario_id=scenario_id,
        input_structure="single-image",
        operation="recognize",
        clinical_target="finding",
        capability_category=CapabilityCategory.RECOGNITION,
        validated_clinical_targets=[{"target_id": "finding", "name": "finding", "target_type": "finding", "subject": "case", "scope": "case", "reference_kind": "direct", "reference_fields": ["label"]}],
        leaf_task_ids=sorted(parsed),
        status="completed",
    )
    return TaskPlanState(dataset={"dataset_id": dataset_id}, scenarios={scenario_id: scenario},
                         leaf_tasks=parsed, phase="complete")


def test_allocation_balances_leaves_within_one_dataset():
    result = allocate_task_plan("a", _plan("a", {"a1": 20, "a2": 20}), 10)
    assert result["allocated_count"] == 10
    assert result["counts"] == {"a": {"a1": 5, "a2": 5}}


def test_allocation_reports_capacity_shortfall_and_zero_leaves():
    result = allocate_task_plan("a", _plan("a", {"a1": 1, "a2": 0}), 4)
    assert result["allocated_count"] == 1
    assert result["shortfall"] == 3
    assert result["counts"]["a"]["a2"] == 0


def test_option_balancing_preserves_answer_and_balances_positions():
    items = [
        {"item_id": str(index), "leaf_task_id": "task", "answer": "yes", "options": ["yes", "no"]}
        for index in range(4)
    ]
    balanced, distribution = balance_option_positions(items, seed=4)
    assert all(item["answer"] in item["options"] for item in balanced)
    assert set(distribution["after"]) == {"task"}
    assert sorted(distribution["after"]["task"].values()) == [2, 2]


def test_candidate_audit_uses_canonical_dataset_binding(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    asset = ImageEntry(path="asset-1", source_id="s", source_digest="raw", kind="image", format="png", source_path="image.png",
                       file_digest="sha256:image", width=4, height=4)
    case = CaseRecord(case_id="case-1", patient_id=None, evidence=[
        Evidence(id="image", type="asset", name="image", value="asset-1"),
        Evidence(id="label", type="value", name="label", value="yes"),
    ])
    (bundle / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    (bundle / "image_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    images = image_context(bundle, [case.model_dump(mode="json")])
    asset = asset.model_copy(update={"file_digest": images.assets["asset-1"]["file_digest"]})
    (bundle / "image_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    _, access = prepare_dataset(bundle / "cases.jsonl", bundle / "image_manifest.jsonl", bundle / "cache")
    plan = _plan("d", {"leaf": 1})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    from medbenchagent.planning.bir import BIRStore
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    store.update("datasets/d/planning/task_plan.json", plan.model_dump(mode="json"), run_id="r", actor="test")

    items = generate_task_items([case.model_dump(mode="json")], plan.leaf_tasks.values(), "d", allocation={"leaf": 1}, resolver_context=images)
    raw = tmp_path / "generated.jsonl"
    write_task_items(raw, items)
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps({"requested_count": 1, "counts": {"d": {"leaf": 1}}}), encoding="utf-8")
    from medbenchagent.agents.finalization import audit_benchmark
    from medbenchagent.data.sources import digest_file
    from medbenchagent.planning.access import DatasetAccessReceipt
    receipt = DatasetAccessReceipt.bind("d", access).model_dump(exclude_none=True)
    store.update("dataset_bindings.json", {"datasets":{"d":{"access":receipt}}}, run_id="r", actor="test")
    store.update("allocation.json", {"requested_count":1, "counts":{"d":{"leaf":1}}}, run_id="r", actor="test")
    store.update("outputs/d.json", {"benchmark_items_path":str(raw), "benchmark_items_digest":digest_file(raw)}, run_id="r", actor="test")
    result = audit_benchmark({"bir_path": str(store.root),
        "selected_dataset_ids": ["d"],
        "allocation_path": str(allocation_path),
        "dataset_task_plans": {"d": str(plan_path)},
        "dataset_generations": {"d": {"raw_output_path": str(raw)}},
        "dataset_assets": {"d": {"access": access}},
        "operation_directory": str(tmp_path),
        "seed": 0,
    }, output_directory=tmp_path / "delivery")
    assert result["updates"]["delivery_counts"]["accepted"] == 1
    assert (tmp_path / "delivery" / "benchmark.jsonl").is_file()
    assert json.loads((tmp_path / "delivery" / "benchmark_manifest.json").read_text())["tasks"]["d::leaf"]["allocated_count"] == 1
