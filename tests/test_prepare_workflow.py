from dataclasses import replace
from pathlib import Path
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from medbenchagent import settings
from medbenchagent.workflow.graph import compile_benchmark_graph
from medbenchagent.data.dataset_catalog import load_dataset_catalog, dataset_by_id, load_prepared
from test_dataset_cards import prepared_dataset


@pytest.mark.parametrize("rebuild,interrupt_prepare", [(False, False), (True, False), (False, True)])
def test_standalone_prepare_registers_complete_card_with_local_backend(tmp_path, monkeypatch, harness_model, rebuild, interrupt_prepare):
    cfg = settings.get_settings()
    cfg = replace(cfg, paths=replace(cfg.paths, dataset_catalog=tmp_path / "catalog.json",
                                    prepared_datasets_root=tmp_path / "prepared", runs_root=tmp_path / "runs"),
                  data_processing={**cfg.data_processing, "backend": "local"},
                  dataset_sources=({"dataset_id": "new", "name": "New", "description": "Status annotations",
                                    "sources": [{"id": "source", "kind": "annotation", "root": str(tmp_path / "input/source")} ]},))
    monkeypatch.setattr(settings, "_SETTINGS", cfg)
    prepared_dataset(tmp_path, "input", monkeypatch)
    previous = prepared_dataset(tmp_path, "new", monkeypatch) if rebuild else None
    card_prompts = []
    def card_model(system, prompt):
        import json
        card_prompts.append(json.loads(prompt))
        assert "Prepare these standard annotations" not in prompt
        assert "concepts" not in json.loads(prompt)
        return {"description": "Dataset facts from Profile"}
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", card_model)
    actions = iter([
        ("prepare_data", {"dataset_id": "new", "mode": "rebuild" if rebuild else "reuse_or_prepare"}),
        ("select_source", {"source_id": "source"}), ("inspect_images", {}),
        ("save_image_plan", {"overrides": {}}), ("normalize_images", {}),
        *([("ask_user", {"question": "Clarify source semantics"})] if interrupt_prepare else []),
        ("save_dataset_metadata", {"source_references": []}),
        ("save_harmonization", {"plan": {}}), ("finish", {}), ("finish", {})])
    def model(*args):
        tool, arguments = next(actions)
        return {"tool": tool, "arguments": arguments, "reason": "Prepare one dataset"}
    harness_model(model)
    graph = compile_benchmark_graph(checkpointer=MemorySaver())
    output = graph.invoke({"user_requirement": "Prepare these standard annotations", "prepare_dataset_id": "new"},
                          {"configurable": {"thread_id": "prepare"}, "recursion_limit": 100})
    if interrupt_prepare:
        assert output["__interrupt__"]
        monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, dataset_sources=()))
        output = graph.invoke(Command(resume="Keep original source semantics"),
                              {"configurable": {"thread_id": "prepare"}, "recursion_limit": 100})
        assert len(list((tmp_path / "prepared/new").iterdir())) == 1
    assert output["status"] == "complete"
    entry = dataset_by_id(load_dataset_catalog(), "new")
    card, access = load_prepared(entry)
    assert card.total_images == 2
    assert Path(access["annotation_path"]).is_relative_to(tmp_path / "prepared/new")
    assert (Path(access["annotation_path"]).parent / "execution_runtime.json").is_file()
    assert len(card_prompts) == 1
    assert card.preparation_metadata.dataset_description is None
    assert (Path(access["annotation_path"]).parent / "dataset_metadata.json").is_file()
    if previous:
        assert entry["annotation_path"] != previous["annotation_path"]
        assert Path(previous["annotation_path"]).is_file()


def test_supervisor_reuses_without_starting_data_processing(tmp_path, monkeypatch, harness_model):
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths,
                        dataset_catalog=tmp_path / "catalog.json", runs_root=tmp_path / "runs")))
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    monkeypatch.setattr("medbenchagent.agents.data_pipeline.initialize", lambda *a: pytest.fail("Must reuse, not prepare"))
    actions = iter([("inspect_dataset_status", {"dataset_id": "a"}), ("prepare_data", {"dataset_id": "a"}), ("finish", {})])
    def model(*args):
        tool, arguments = next(actions)
        return {"tool": tool, "arguments": arguments, "reason": "Reuse"}
    harness_model(model)
    output = compile_benchmark_graph().invoke({"user_requirement": "Prepare if needed", "prepare_dataset_id": "a"})
    assert output["status"] == "complete"
    assert output["prepared_entries"]["a"]["annotation_path"] == entry["annotation_path"]


def test_confirmed_intake_prepares_and_finishes_without_supervisor_question(tmp_path, monkeypatch, harness_model):
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths,
                        dataset_catalog=tmp_path / "catalog.json", runs_root=tmp_path / "runs")))
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    confirmed_root = tmp_path / "confirmed"
    confirmed_root.mkdir()
    harness_model(lambda *_args: pytest.fail("Confirmed intake must not invoke Supervisor decisions"))
    output = compile_benchmark_graph().invoke({"user_requirement": "Prepare A",
                                               "prepare_dataset_id": "a", "intake_dataset_name": "A",
                                               "intake_source_root": str(confirmed_root)})
    assert output["status"] == "complete"
    assert output["prepared_entries"]["a"]["annotation_path"] == entry["annotation_path"]


def test_prepare_discovers_unconfigured_source_from_request(tmp_path, monkeypatch, harness_model):
    cfg = settings.get_settings()
    cfg = replace(cfg, paths=replace(cfg.paths, dataset_catalog=tmp_path / "catalog.json",
                                     prepared_datasets_root=tmp_path / "prepared",
                                     runs_root=tmp_path / "runs"),
                  data_processing={**cfg.data_processing, "backend": "local"}, dataset_sources=())
    monkeypatch.setattr(settings, "_SETTINGS", cfg)
    prepared_dataset(tmp_path, "input", monkeypatch)
    source = tmp_path / "input" / "source"
    monkeypatch.setattr("medbenchagent.agents.source_discovery.complete_json",
                        lambda *_a: {"root": str(source), "dataset_name": "New"})
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json",
                        lambda *_a: {"description": "Dataset facts from Profile"})
    actions = iter([
        ("prepare_data", {"dataset_id": "new"}),
        ("select_source", {"source_id": "original"}), ("inspect_images", {}),
        ("save_image_plan", {"overrides": {}}), ("normalize_images", {}),
        ("save_dataset_metadata", {"source_references": []}),
        ("save_harmonization", {"plan": {}}), ("finish", {}), ("finish", {}),
    ])
    harness_model(lambda *_a: {"tool": (action := next(actions))[0],
                               "arguments": action[1], "reason": "Prepare dataset"})
    output = compile_benchmark_graph(checkpointer=MemorySaver()).invoke(
        {"user_requirement": f"Prepare New from {source}", "prepare_dataset_id": "new"},
        {"configurable": {"thread_id": "discover"}, "recursion_limit": 100})
    assert output["status"] == "complete"
    assert (Path(output["run_directory"]) / ".runtime/discovery/sources/new/source_decision.json").is_file()


def test_preparation_tools_follow_existing_artifact_dependencies(tmp_path):
    from medbenchagent.agents.data_pipeline import available_tools
    from medbenchagent.runtime.dependencies import publish
    c = {"run_directory": str(tmp_path), "sources": [{"id": "one"}], "dataset_scope": "dataset"}
    assert "select_source" in available_tools(c)
    assert "inspect_images" not in available_tools(c)
    c["source_scope"] = "one"
    publish(c, "Source", {}, ())
    assert "inspect_files" in available_tools(c)
    publish(c, "Annotation", {}, ("Source",))
    assert "inspect_images" in available_tools(c)
    assert "normalize_images" not in available_tools(c)
    publish(c, "ImageInventory", {}, ("Annotation",))
    publish(c, "ImagePlan", {}, ("ImageInventory",))
    assert "normalize_images" in available_tools(c)
    publish(c, "Annotation", {"revision": 2}, ("Source",))
    assert "normalize_images" not in available_tools(c)
