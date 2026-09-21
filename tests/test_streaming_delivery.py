import json
from pathlib import Path

import pytest

from medbenchagent.audit import assembly, storage
from medbenchagent.audit.hard import inspect_generation, read_rows, write_rows
from medbenchagent.data.dataset_catalog import load_prepared
from medbenchagent.data.query import DatasetQuery
from medbenchagent.planning.protocol import template_specification
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.run_artifacts import atomic_write_json
from test_dataset_cards import prepared_dataset
from test_generation import _task


@pytest.mark.parametrize("total_limit,expected", [(1, 2), (None, 3)])
def test_streaming_delivery_cross_batch_conflicts_limits_and_reconciliation(tmp_path, monkeypatch, total_limit, expected):
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    _, access = load_prepared(entry)
    image_ids = [row["image"]["image_id"] for row in read_rows(Path(entry["annotation_path"]))]
    monkeypatch.setattr(storage, "BATCH_SIZE", 1)
    original_records = DatasetQuery.records
    def bounded_records(self, keys):
        assert len(keys) <= 1
        return original_records(self, keys)
    monkeypatch.setattr(DatasetQuery, "records", bounded_records)
    monkeypatch.setattr("medbenchagent.data.validation.load_annotation_jsonl", lambda *a: pytest.fail("Full annotation load"))
    monkeypatch.setattr("medbenchagent.data.image_assets.load_image_manifest", lambda *a: pytest.fail("Full manifest load"))
    tasks = {key: _task("letter", ["image.annotation.status"]) for key in ("first", "second")}
    for task in tasks.values():
        task.pop("question_templates")
    plan = {"dataset": {"dataset_id": "a"}, "benchmark_info": {"language": "en", "sample_count": 4},
            "concepts": {"concept": {"description": "Status", "category": "test", "origin": "core"}}, "tasks": tasks}
    templates = {"templates": {key: ["Status?", "Which status?", "Identify status."] for key in tasks},
                 "reviews": {key: {"specification": template_specification(task["protocol"], "en").model_dump(),
                                    "review": {"verdict": "approved"}} for key, task in tasks.items()}}
    plan_path, templates_path, raw = tmp_path / "plan.json", tmp_path / "templates.json", tmp_path / "raw.jsonl"
    atomic_write_json(plan_path, plan)
    atomic_write_json(templates_path, templates)
    def item(leaf, image, question="Status?"):
        return {"id": f"{leaf}-{image}", "dataset": "a", "task_id": leaf, "image_id": image_ids[image],
                "image_path": f"{image}.png", "question": question, "options": ["benign", "malignant"],
                "answer": "A" if image == 0 else "B"}
    write_rows(raw, [item("first", 0), item("first", 1), item("second", 0), item("first", 0),
                     item("first", 1, "Which status?"), item("second", 1)])
    with raw.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    checked = inspect_generation(dataset_id="a", raw_path=raw, plan_path=plan_path, templates_path=templates_path,
                                 dataset_access=access, manifest_path=Path(entry["image_manifest_path"]), output_directory=tmp_path / "check")
    allocation = tmp_path / "allocation.json"
    atomic_write_json(allocation, {"counts": {"a": {"first": 2, "second": 2}}, "requested_count": 4, "allocated_count": 4})
    context = {"run_id": "stream", "run_directory": str(tmp_path), "selected_dataset_ids": ["a"],
               "benchmark_info": plan["benchmark_info"], "dataset_task_plans": {"a": str(plan_path)},
               "dataset_templates": {"a": str(templates_path)}, "dataset_assets": {"a": {"entry": entry, "access": access}},
               "dataset_checks": {"a": checked}, "dataset_generations": {"a": {"raw_output_path": str(raw)}},
               "allocation_path": str(allocation), "max_tasks_per_image": 2, "max_total_tasks_per_image": total_limit}
    for kind in ("DatasetSelection", "Allocation", "FinalizationConfig", "dataset/a/Plan", "dataset/a/Templates", "dataset/a/GenerationCheck"):
        publish(context, kind, {}, ())
    root = assembly.assemble(context, tmp_path / "assembly")
    assembly.balance(root, 3)
    assert assembly.validate_package(context, root)["item_count"] == expected
    report = json.loads((root / "package/audit_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["generated"] == 7
    assert report["counts"]["total_shortfall"] == 4 - expected
    assert len(list(read_rows(root / "quarantine.jsonl"))) == 7 - expected
    assert report["rejection_reason_counts"]["conflicting_identity"] == 2
    assert report["rejection_reason_counts"]["duplicate_identity"] == 1
    # Rebuilding disposable indexes and balancing produces the same package rows.
    before = (root / "package/benchmark.jsonl").read_bytes()
    assembly.assemble(context, root)
    assembly.balance(root, 3)
    assert (root / "package/benchmark.jsonl").read_bytes() == before
    if total_limit is None:
        context["max_total_tasks_per_image"] = 1
        with pytest.raises(ValueError, match="per-image limits"):
            assembly.validate_package(context, root)
    else:
        rows = list(read_rows(root / "package/benchmark.jsonl"))
        write_rows(root / "package/benchmark.jsonl", rows + [rows[0]])
        with pytest.raises(ValueError, match="identities must be unique"):
            assembly.validate_package(context, root)
