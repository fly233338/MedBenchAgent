import json
import shutil
from pathlib import Path

import pytest
from compilation_fixtures import compile_fixture
from test_global_construction import _plan
from medbenchagent.agents import task_planning
from medbenchagent.agents.review import review_plan
from medbenchagent.generation.items import run_generation
from medbenchagent.evaluation.loader import load_benchmark_package
from medbenchagent.generation.delivery import publish_benchmark, export_benchmark
from medbenchagent.planning.bir import BIRStore, BIRStaleError
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime import sessions


def cases():
    return [{"case_id": "case", "evidence": [
        {"id": "image", "type": "asset", "name": "image", "value": "image.png", "ref": []},
        {"id": "label", "type": "value", "name": "label", "value": "yes", "ref": []},
    ]}]


def approve(context, monkeypatch):
    def model(_prompt, payload, _schema):
        value = json.loads(payload)
        assert value["source_evidence"]["representative_cases"]
        return {"scenario_id": value["scenario"]["scenario_id"],
                "checked_leaf_task_ids": [leaf["leaf_task_id"] for leaf in value["leaf_tasks"]],
                "status": "approved"}
    monkeypatch.setattr(task_planning, "complete_json", model)
    assert review_plan(context)["updates"]["review_summary"]["status"] == "approved"


def test_canonical_stages_and_portable_export_survive_relocation(tmp_path, monkeypatch):
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    candidate = Path(context["candidate_path"])
    assert not (candidate / "complete.json").exists()
    with pytest.raises((ValueError, OSError)):
        publish_benchmark(context)
    approve(context, monkeypatch)
    delivered = publish_benchmark(context)["updates"]["delivery_path"]
    original = load_benchmark_package(delivered)
    assert original["items"][0]["leaf_task_id"] == "leaf"
    exported = Path(export_benchmark(delivered, tmp_path / "export"))
    assert export_benchmark(delivered, exported) == str(exported)
    moved = tmp_path / "moved"
    exported.rename(moved)
    shutil.rmtree(tmp_path / "dataset")
    shutil.rmtree(tmp_path / "BIR")
    shutil.rmtree(tmp_path / "runtime")
    portable = load_benchmark_package(moved)
    assert portable["manifest"]["benchmark_id"] == original["manifest"]["benchmark_id"]
    assert Path(portable["items"][0]["input_images"][0]["path"]).is_relative_to(moved)


def test_source_image_mutation_is_rejected_before_export(tmp_path, monkeypatch):
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    approve(context, monkeypatch)
    delivered = publish_benchmark(context)["updates"]["delivery_path"]
    original_image = (tmp_path / "dataset/image.png").read_bytes()
    (tmp_path / "dataset/image.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        export_benchmark(delivered, tmp_path / "export")
    (tmp_path / "dataset/image.png").write_bytes(original_image)
    assert load_benchmark_package(export_benchmark(delivered, tmp_path / "export"))["items"]


def test_revised_plan_invalidates_allocation_generation_and_review(tmp_path, monkeypatch):
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    approve(context, monkeypatch)
    from medbenchagent.workflow.supervision import revise_plan
    revise_plan(context, scenario_ids=["scenario-d"], reason="Check boundary wording")
    store = BIRStore(context["bir_path"])
    for path in ("allocation.json", "review.json", "candidate.json"):
        with pytest.raises(BIRStaleError):
            store.load(path)
    with pytest.raises(BIRStaleError):
        run_generation(context)


def test_scoped_rework_keeps_frozen_extension_dependency(tmp_path):
    from medbenchagent.workflow.supervision import revise_plan
    from medbenchagent.planning.extension_selection import select_extensions
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    revise_plan(context, scenario_ids=["scenario-d"], reason="Check wording")
    select_extensions(context, extension_ids=[], selection_reason={})
    with pytest.raises(BIRStaleError, match="planning_extensions.json"):
        BIRStore(context["bir_path"]).load("task_plan.json")


def test_batch_has_independent_business_state_and_frozen_config(tmp_path):
    sessions.create_experiment("pilot", root=tmp_path)
    service = ConversationService("pilot", root=tmp_path)
    submitted = service.submit_batch(["a", "b"])
    first, second = service.record()["runs"]
    assert len(submitted.run_ids) == 2
    assert first["bir_path"] != second["bir_path"]
    BIRStore(first["bir_path"]).update("benchmark_info.json", {"sample_count": 3}, run_id="r", actor="test")
    assert BIRStore(second["bir_path"]).load("benchmark_info.json")["sample_count"] is None
    assert "bir_path" not in service.record()


def test_supervisor_drives_independent_review_publication_and_finish(tmp_path, monkeypatch, harness_model):
    import medbenchagent.workflow.graph as workflow
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 1}), cases())
    context.update(user_requirement="Compile d",
                   construction_events_path=str(tmp_path / "runtime/events.jsonl"))
    monkeypatch.setattr(workflow, "complete_validated_json", lambda *args, **kwargs:
                        workflow.SupervisorRequestIntent(goal="benchmark"))
    def reviewer(_prompt, payload, _schema):
        value = json.loads(payload)
        return {"scenario_id": value["scenario"]["scenario_id"],
                "checked_leaf_task_ids": [leaf["leaf_task_id"] for leaf in value["leaf_tasks"]], "status": "approved"}
    monkeypatch.setattr(task_planning, "complete_json", reviewer)
    actions = iter(["inspect_compilation", "review_plan", "publish", "finish"])
    harness_model(lambda *args: {"tool": next(actions), "arguments": {}, "reason": "Evaluate evidence and deliver"})
    captured = {}
    original = workflow.compile_agent
    def capture(*args, **kwargs):
        graph = original(*args, **kwargs)
        captured[args[0]] = graph
        return graph
    monkeypatch.setattr(workflow, "compile_agent", capture)
    workflow.compile_benchmark_graph()
    outcome = captured["SupervisorAgent"].invoke({"context": context})
    assert outcome["result"]["status"] == "complete"
    assert load_benchmark_package(outcome["context"]["delivery_path"])["items"]
    assert outcome["context"]["view"]["review"]["status"] == "approved"


def test_preparation_only_supervisor_publishes_once_with_one_child_reference(tmp_path, monkeypatch):
    import medbenchagent.workflow.graph as workflow
    from medbenchagent.preparation import entrypoint
    from medbenchagent.preparation.graph import compile_preparation_graph
    from medbenchagent.runtime.contexts import load_context
    from test_preparation_graph import Worker, state
    initial = state(tmp_path)
    worker = Worker()
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    source = {"dataset_id": "d", "sources": initial["sources"]}
    monkeypatch.setattr(workflow, "complete_validated_json", lambda *args, **kwargs:
                        workflow.SupervisorRequestIntent(goal="preparation_only", preparation_dataset_id="d"))
    monkeypatch.setattr(workflow, "configured_source", lambda _dataset: source)
    monkeypatch.setattr(workflow, "dataset_status", lambda *_args: {"prepared": False, "available": None, "source_errors": []})
    monkeypatch.setattr(entrypoint, "_coding_worker_identity", lambda: {})
    def prepare(_context):
        prepared = compile_preparation_graph(worker=worker).invoke(initial)
        return {"status": prepared["status"], "summary": "Prepared", "updates": {"publication": prepared["publication"]}}
    monkeypatch.setattr(entrypoint, "run_preparation_agent", prepare)
    captured = {}
    original = workflow.compile_agent
    def capture(*args, **kwargs):
        graph = original(*args, **kwargs)
        captured[args[0]] = graph
        return graph
    monkeypatch.setattr(workflow, "compile_agent", capture)
    workflow.compile_benchmark_graph()
    context = {"bir_path": str(store.root), "run_id": "r", "run_directory": initial["run_directory"],
               "operation_directory": str(tmp_path / "operations"),
               "construction_events_path": str(tmp_path / "events.jsonl"),
               "registration_catalog_path": str(tmp_path / "catalog.json"),
               "user_requirement": f"Prepare dataset d from {initial['source_roots']['s']}"}
    outcome = captured["SupervisorAgent"].invoke({"context": context})
    assert outcome["result"]["status"] == "complete"
    assert outcome["context"]["prepared_entry"]["dataset_id"] == "d"
    assert set(outcome["context"]["child_states"]) == {"preparation"}
    saved = load_context(outcome["context"]["child_states"]["preparation"], "preparation", "d")
    assert "access_manifest" not in saved["publication"]
    assert "dataset_access_manifest" not in saved
    assert store.current("task_plan.json") is None
    calls = len(worker.requests)
    captured["SupervisorAgent"].invoke({"context": outcome["context"]})
    assert len(worker.requests) == calls
