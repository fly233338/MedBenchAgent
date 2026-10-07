import json
from pathlib import Path

import pytest

from medbenchagent.planning.bir import BIRStore
from medbenchagent.runtime import sessions


def test_restore_business_preserves_current_run_request(tmp_path, monkeypatch):
    from medbenchagent.planning import bir

    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot/BIR")
    store.update("benchmark_info.json", {"schema_version": "benchmark-info", "user_requirement": "Build old benchmark"},
                 run_id="old", actor="test")
    monkeypatch.setattr(bir, "require", lambda _context, kind: {kind: {}} if kind == "Requirement" else None)
    context = {"bir_path": str(store.root), "user_requirement": "Prepare new dataset"}
    bir.restore_business(context)
    assert context["user_requirement"] == "Prepare new dataset"
    assert context["brief_requirement"] == "Build old benchmark"


def test_new_experiment_initializes_bir_directory_and_manifest(tmp_path: Path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    bir = tmp_path / "pilot" / "BIR"
    assert Path(record["bir_path"]) == bir
    assert (bir / "registry" / "axes.json").is_file()
    manifest = json.loads((bir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bir"
    assert "revision" not in manifest and "parts" not in manifest
    assert record["bir_manifest_digest"].startswith("sha256:")


def test_bir_revision_conflict_and_digest_validation(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    first = store.update("benchmark_info.json", {"schema_version": "benchmark-info", "value": 1},
                         run_id="r1", actor="test", expected_revision=0)
    assert first.revision == 1
    with pytest.raises(ValueError, match="revision conflict"):
        store.update("benchmark_info.json", {"schema_version": "benchmark-info", "value": 2},
                     run_id="r2", actor="test", expected_revision=0)
    assert store.validate().valid
    (tmp_path / "pilot" / "BIR" / "benchmark_info.json").write_text("{}", encoding="utf-8")
    assert not store.validate().valid


def test_bir_rejects_path_escape(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    with pytest.raises(ValueError):
        store.load("../outside.json")


def test_task_plan_is_only_planning_authority(tmp_path):
    from medbenchagent.agents import task_planning
    from medbenchagent.planning.task_space import TaskPlanState
    record = sessions.create_experiment("pilot", root=tmp_path)
    operation = tmp_path / "operation"
    context = {"bir_path": record["bir_path"], "dataset_scope": "demo", "run_id": "r",
               "operation_directory": str(operation), "dataset_access_manifest": {"revision": "r"}}
    BIRStore(record["bir_path"]).update("dataset_bindings.json", {"datasets": {"demo": {"access": {"revision": "r"}}}}, run_id="r", actor="test")
    plan = TaskPlanState(dataset={"dataset_id": "demo"})
    result = task_planning._write_artifacts(context, plan)
    assert Path(result["task_plan_path"]).is_relative_to(Path(record["bir_path"]))
    assert list((Path(record["bir_path"]) / "datasets/demo/planning").glob("*.json")) == [Path(result["task_plan_path"])]
    (operation / "task_plan.json").write_text("corrupted", encoding="utf-8")
    assert not (operation / "leaf_tasks.json").exists()
    recovered = task_planning._load_plan(context)
    assert recovered == plan
    task_planning._write_artifacts(context, recovered)
    assert not (operation / "leaf_tasks.json").exists()


def test_bir_json_is_readable_and_failed_transaction_preserves_previous_file(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    store.update("allocation.json", {"schema_version": "allocation", "items": ["one"]},
                 run_id="r1", actor="test")
    path = store.root / "allocation.json"
    text = path.read_text(encoding="utf-8")
    assert "\n  \"items\"" in text
    before = json.loads(text)
    with pytest.raises(TypeError):
        store.update("allocation.json", {"schema_version": "allocation", "items": {1, 2}},
                     run_id="r2", actor="test")
    assert json.loads(path.read_text(encoding="utf-8")) == before



def test_atomic_update_failure_and_dependency_invalidation(tmp_path, monkeypatch):
    from medbenchagent.planning import bir
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot/BIR")
    first = store.update("benchmark_info.json", {"scope":"one"}, run_id="r", actor="test")
    assert store.update("benchmark_info.json", {"scope":"one"}, run_id="r", actor="test").revision == first.revision
    store.update("allocation.json", {"count":1}, run_id="r", actor="test",
        dependencies={"benchmark_info.json":store.digest("benchmark_info.json")})
    replace = bir.os.replace
    def interrupted(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(bir.os, "replace", interrupted)
    with pytest.raises(OSError):
        store.update("benchmark_info.json", {"scope":"two"}, run_id="r", actor="test")
    assert store.load("benchmark_info.json") == {"scope":"one"}
    assert store.load("allocation.json") == {"count":1}
    monkeypatch.setattr(bir.os, "replace", replace)
    store.update("benchmark_info.json", {"scope":"two"}, run_id="r", actor="test")
    with pytest.raises(ValueError, match="dependency changed"):
        store.load("allocation.json")
