from compilation_fixtures import registry, planning_context
import json

import pytest

from medbenchagent.data.dataset_catalog import normalize_dataset_id
from medbenchagent.data import source_discovery
from medbenchagent.workflow.graph import automatic_supervisor_action, available_stages
from medbenchagent.workflow import graph as workflow


def test_dynamic_dataset_id_uses_directory_name(tmp_path):
    root = tmp_path / "BrEaST Dataset"
    root.mkdir()

    assert source_discovery.dynamic_dataset_id(f'Prepare "{root}"') == "breast-dataset"
    assert source_discovery.dynamic_dataset_id("Prepare BrEaST") is None


def test_normalize_dataset_id_lowercases_names():
    assert normalize_dataset_id("BrEaST") == "breast"


def test_catalog_loading_does_not_require_redundant_image_manifest_field(tmp_path):
    from medbenchagent.data.dataset_catalog import load_dataset_catalog

    path = tmp_path / "catalog.json"
    entry = {"dataset_id": "breast", "available": True, "cases_path": "cases.jsonl"}
    path.write_text(json.dumps({"datasets": [entry]}), encoding="utf-8")
    assert load_dataset_catalog(path) == [entry]


def test_discover_source_uses_explicit_path_without_llm(tmp_path, monkeypatch):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")

    def fail(*_args, **_kwargs):
        raise AssertionError("explicit paths must not invoke source-discovery LLM")

    monkeypatch.setattr(source_discovery, "complete_validated_json", fail)
    entry = source_discovery.discover_source(
        {"run_directory": str(tmp_path / "run"), "user_requirement": f"Prepare {root}"},
        "breast",
    )

    assert entry["dataset_id"] == "breast"
    assert entry["sources"][0]["root"] == str(root.resolve())
    decision = json.loads((tmp_path / "run" / "discovery" / "source_decision.json").read_text())
    assert decision["entry"] == entry


def test_discover_source_does_not_cache_conflicting_registration(tmp_path, monkeypatch):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")

    def fail(*_args, **_kwargs):
        raise ValueError("Conflicting dataset_sources entry: breast")

    monkeypatch.setattr("medbenchagent.runtime.sessions.upsert_dataset_source", fail)
    with pytest.raises(ValueError, match="Conflicting dataset_sources"):
        source_discovery.discover_source(
            {"run_directory": str(tmp_path / "run"), "experiments_root": str(tmp_path),
             "experiment_name": "pilot", "user_requirement": f"Prepare {root}"},
            "breast",
        )
    assert not (tmp_path / "run" / "discovery" / "source_decision.json").exists()


def test_prepare_only_routes_directly_and_finishes_without_benchmark_fields(tmp_path, monkeypatch):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")
    context = {"run_directory": str(tmp_path / "run"), "user_requirement": f"Prepare {root}"}
    monkeypatch.setattr(workflow, "complete_validated_json", lambda _prompt, _request, schema:
                        schema(goal="preparation_only", preparation_dataset_id="breast"))
    context = workflow.initialize_supervisor_intent(context)

    action = automatic_supervisor_action(context)
    assert action["tool"] == "process_dataset"
    assert action["arguments"] == {"dataset_id": "breast", "mode": "reuse_or_process"}
    assert {"ask_user", "process_dataset"} <= available_stages(context)
    assert "specify" not in available_stages(context)
    assert "discover_datasets" not in available_stages(context)
    context["prepared_entry"] = {"dataset_id": "breast"}
    assert automatic_supervisor_action(context)["tool"] == "finish"
    assert available_stages(context) == {"finish"}
    assert workflow.finish_supervisor(context)["status"] == "complete"
    assert "process_dataset" in registry().prompt("supervisor")
    assert "Preparation-only requests finish after validated registration" in registry().prompt("supervisor")
    assert "Resolve an explicitly named dataset with bind_dataset" in registry().prompt("supervisor")
    assert "请" not in registry().prompt("supervisor")


@pytest.mark.parametrize("prepare", [False, True])
def test_benchmark_and_joint_requests_require_benchmark_flow(tmp_path, monkeypatch, prepare):
    root = tmp_path / "BrEaST"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")
    requirement = (f"Prepare dataset BrEaST from {root} and then use it for a Chinese benchmark"
                   if prepare else "Build a Chinese benchmark using dataset BrEaST")
    monkeypatch.setattr(workflow, "complete_validated_json", lambda _prompt, _request, schema:
                        schema(goal="benchmark", preparation_dataset_id="breast" if prepare else None))
    context = workflow.initialize_supervisor_intent({**planning_context(tmp_path), "user_requirement": requirement})
    action = automatic_supervisor_action(context)
    assert (action["tool"] if action else None) == ("process_dataset" if prepare else None)
    context["prepared_entry"] = {"dataset_id": "breast"}
    assert automatic_supervisor_action(context) is None
    assert "specify" in available_stages(context)
    assert "finish" not in available_stages(context)
    with pytest.raises(ValueError, match="Delivery is not complete"):
        workflow.finish_supervisor(context)


def test_preparation_clarification_and_requested_dataset_completion(monkeypatch):
    context = {"user_requirement": "Prepare dataset X",
               "request_intent": {"goal": "preparation_only", "preparation_dataset_id": "x"},
               "prepared_entry": {"dataset_id": "other"}}
    assert "finish" not in available_stages(context)
    with pytest.raises(ValueError, match="requested dataset"):
        workflow.finish_supervisor(context)
    context["global_issue"] = {"status": "pending", "issue_id": "semantic", "question": "Which scope?"}
    assert available_stages(context) == {"ask_user"}
    assert automatic_supervisor_action(context) is None
    context["global_issue"]["status"] = "answered"
    assert automatic_supervisor_action(context)["tool"] == "resolve_clarification"


def test_request_intent_is_reused_on_resume(monkeypatch):
    context = {"user_requirement": "Prepare X", "intent_requirement": "Prepare X",
               "request_intent": {"goal": "preparation_only", "preparation_dataset_id": "x"}}
    monkeypatch.setattr(workflow, "complete_validated_json", lambda *_args: pytest.fail("unchanged intent reclassified"))
    assert workflow.initialize_supervisor_intent(context) == context


def test_preparation_only_harness_never_calls_specify_or_asks_language(tmp_path, monkeypatch):
    from medbenchagent.runtime.harness import compile_agent, result

    root = tmp_path / "raw"
    root.mkdir()
    (root / "image.png").write_bytes(b"pixels")
    calls = []
    monkeypatch.setattr(workflow, "complete_validated_json", lambda _prompt, _request, schema:
                        schema(goal="preparation_only", preparation_dataset_id="raw"))
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_validated_json",
                        lambda *_args, **_kwargs: pytest.fail("No Supervisor model action or benchmark question expected"))

    def process(context, **arguments):
        calls.append(arguments)
        return result("success", "Validated and registered", prepared_entry={"dataset_id": "raw"})

    graph = compile_agent(
        "SupervisorAgent", workflow._supervisor_instructions,
        {"process_dataset": process, "specify": lambda _: pytest.fail("specify called")},
        lambda _: {}, workflow.finish_supervisor,
        available_tools=workflow.available_stages,
        automatic_action=workflow.automatic_supervisor_action,
        initialize_context=workflow.initialize_supervisor_intent,
        args_schemas={"process_dataset": workflow.ProcessDatasetArguments},
    )
    output = graph.invoke({"context": {
        "run_id": "test", "run_directory": str(tmp_path / "run"),
        "operation_directory": str(tmp_path / "run/ops"),
        "construction_events_path": str(tmp_path / "events.jsonl"),
        "user_requirement": f"Prepare dataset raw from {root}",
    }})
    assert output["result"]["status"] == "complete"
    assert len(calls) == 1
    assert calls[0]["dataset_id"] == "raw"


def test_source_discovery_still_reports_unusable_path(tmp_path):
    root = tmp_path / "BrEaST"
    root.mkdir()

    error = source_discovery.explicit_source_error(f"Prepare {root}")
    assert error is not None
    assert "empty" in error


def test_supervisor_has_one_binding_entry_after_specification(tmp_path):
    from medbenchagent.planning.bir import BIRStore
    store = BIRStore.initialize(tmp_path / "BIR", "config/bir_template", "test")
    store.update("benchmark_info.json", {"user_requirement": "Build a benchmark"}, run_id="r", actor="test")
    stages = available_stages({"bir_path": str(store.root), "user_requirement": "Build a benchmark"})
    assert {"discover_datasets", "bind_dataset"} <= stages
    assert "select_datasets" not in stages


def test_stalled_planning_does_not_offer_plan(tmp_path):
    context = {**planning_context(tmp_path), "planning_stalled": True}
    stages = available_stages(context)
    assert "plan" not in stages
    assert "ask_user" in stages
