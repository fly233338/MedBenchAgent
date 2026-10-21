import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from compilation_fixtures import compile_fixture
from test_compilation import cases
from test_global_construction import _plan
from medbenchagent.audit.task_items import audit_task_items
from medbenchagent.contracts import BenchmarkItem
from medbenchagent.generation.items import write_task_items
from medbenchagent.data.contracts import DatasetAccessReceipt
from medbenchagent.planning.bir import BIRStore


@pytest.fixture
def compiled(tmp_path):
    return compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())


def test_canonical_item_boundary_rejects_aliases_and_old_only_items(compiled, tmp_path):
    store = BIRStore(compiled["bir_path"])
    output = store.load("generation.json")
    item = json.loads(Path(output["benchmark_items_path"]).read_text(encoding="utf-8"))
    assert not {"id", "dataset", "task_id"} & item.keys()
    BenchmarkItem.model_validate(item)
    for canonical, alias in (("item_id", "id"), ("dataset_id", "dataset"), ("leaf_task_id", "task_id")):
        for keep_canonical in (True, False):
            old = {**item, alias: item[canonical]}
            if not keep_canonical:
                del old[canonical]
            with pytest.raises(ValidationError):
                write_task_items(tmp_path / "old.jsonl", [old])
            report = audit_task_items(cases(), [old], _plan("d", {"leaf": 1}).leaf_tasks.values())
            assert report["reason_counts"] == {"invalid_item_contract": 1}
    assert not (tmp_path / "old.jsonl").exists()


def test_receipt_checks_identity_counts_and_content(compiled):
    store = BIRStore(compiled["bir_path"])
    value = store.load("dataset_binding.json")["access"]
    receipt = DatasetAccessReceipt.model_validate(value)
    assert receipt.case_count == 1
    access = receipt.verify().model_dump(exclude_none=True)
    with pytest.raises(ValueError, match="counts differ"):
        receipt.model_copy(update={"case_count": 0}).verify()
    Path(access["cases_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        receipt.verify().model_dump(exclude_none=True)


def test_receipt_rejects_removed_nested_access(compiled):
    value = BIRStore(compiled["bir_path"]).load("dataset_binding.json")["access"]
    value["source_access"] = dict(value)
    with pytest.raises(ValidationError, match="Extra inputs"):
        DatasetAccessReceipt.model_validate(value)


def test_all_legacy_items_are_quarantined_without_publishing(compiled, tmp_path):
    from medbenchagent.audit.task_items import audit_benchmark
    from medbenchagent.data.sources import digest_file
    store = BIRStore(compiled["bir_path"])
    output = store.load("generation.json")
    item = json.loads(Path(output["benchmark_items_path"]).read_text(encoding="utf-8"))
    item["task_id"] = item.pop("leaf_task_id")
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps(item) + "\n", encoding="utf-8")
    store.update("generation.json", {**output, "benchmark_items_path": str(legacy),
                                   "benchmark_items_digest": digest_file(legacy)}, run_id="test", actor="test")
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="No deliverable") as error:
        audit_benchmark(compiled, output_directory=rejected)
    assert error.value.diagnostics["reason_counts"] == {"invalid_item_contract": 1}
    assert json.loads((rejected / "quarantine.jsonl").read_text())["reasons"] == ["invalid_item_contract"]
    assert not (rejected / "complete.json").exists()


def test_audit_rejects_non_object_records_without_crashing():
    report = audit_task_items(cases(), [None, [], 42], _plan("d", {"leaf": 1}).leaf_tasks.values())
    assert report["rejected"] == 3
    assert report["reason_counts"] == {"invalid_item_contract": 3}


def test_planning_cache_cannot_bypass_current_access_integrity(tmp_path):
    from compilation_fixtures import planning_context
    from medbenchagent.agents import task_planning
    context = planning_context(tmp_path)
    access = BIRStore(context["bir_path"]).load("dataset_binding.json")["access"]
    context["dataset_facts"] = {"revision": access["revision"], "case_count": 1}
    Path(access["cases_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        task_planning.read_dataset_overview(context)
