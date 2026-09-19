"""Export validates local provenance and does not append duplicate deliveries."""

import json
from pathlib import Path
import pytest
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.run_export import export_benchmark
from test_dataset_cards import prepared_dataset
from test_generation import _task


def export_context(tmp_path, monkeypatch):
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    root = tmp_path / "run"
    root.mkdir()
    manifest = json.loads(Path(entry["image_manifest_path"]).read_text().splitlines()[0])
    leaf = _task("letter", ["image.annotation.status"])
    leaf.update(eligible_count=2)
    leaf.pop("question_templates")
    leaf["concept_id"] = "core"
    plan = {"tasks": {"leaf": leaf}, "concepts": {"core": {"origin": "core", "description": "Status", "category": "test"}}}
    plan_path = root / "plan.json"
    atomic_write_json(plan_path, plan)
    allocation = {"counts": {"a": {"leaf": 1}}, "requested_count": 2, "allocated_count": 1, "planning_shortfall": 1}
    atomic_write_json(root / "allocation.json", allocation)
    row = {"id": "old", "task_id": "leaf", "dataset": "a", "image_id": manifest["image_id"],
           "image_path": manifest["png_path"], "question": "Status?", "options": ["Benign", "Malignant"], "answer": "A"}
    for name in ("raw", "audited"):
        (root / f"local-{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    atomic_write_json(root / "local-report.json", {"hard_audit": {"accepted": 1, "rejected": 0}})
    c = {"run_id": "test", "run_directory": str(root), "benchmark_info": {"sample_count": 2},
         "selected_dataset_ids": ["a"], "allocation_path": str(root / "allocation.json"),
         "dataset_task_plans": {"a": str(plan_path)}, "dataset_assets": {"a": {"entry": entry}},
         "dataset_generations": {"a": {"raw_output_path": str(root / "local-raw.jsonl")}},
         "dataset_audits": {"a": {"audited": str(root / "local-audited.jsonl"), "report": str(root / "local-report.json")}}}
    for kind in ("DatasetSelection", "Allocation", "dataset/a/Audit"):
        publish(c, kind, {}, ())
    return c


def test_export_replay_and_counts(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch)
    first = export_benchmark(c)
    data = Path(first["audit_artifacts"]["audited"]).read_bytes()
    assert first["item_count"] == 1 and first["total_shortfall"] == 1
    again = export_benchmark(c)
    assert Path(again["audit_artifacts"]["audited"]).read_bytes() == data
    assert first["artifact_versions"]["Export"] == again["artifact_versions"]["Export"]


@pytest.mark.parametrize("field,value", [("dataset", "b"), ("task_id", "missing"), ("image_id", "missing")])
def test_export_rejects_wrong_source_references(tmp_path, monkeypatch, field, value):
    c = export_context(tmp_path, monkeypatch)
    path = Path(c["dataset_audits"]["a"]["audited"])
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="references"):
        export_benchmark(c)
    assert not (Path(c["run_directory"]) / "benchmark_manifest.json").exists()


def test_export_rejects_duplicate_samples(tmp_path, monkeypatch):
    c = export_context(tmp_path, monkeypatch)
    path = Path(c["dataset_audits"]["a"]["audited"])
    path.write_text(path.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        export_benchmark(c)
