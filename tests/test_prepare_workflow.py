from dataclasses import replace
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from medbenchagent import settings
from medbenchagent.workflow.graph import compile_benchmark_graph
from medbenchagent.data.dataset_catalog import load_dataset_catalog, dataset_by_id, load_prepared
from test_dataset_cards import prepared_dataset


def test_standalone_prepare_registers_complete_card_with_local_backend(tmp_path, monkeypatch):
    cfg = settings.get_settings()
    cfg = replace(cfg, paths=replace(cfg.paths, dataset_catalog=tmp_path / "catalog.json",
                                    prepared_datasets_root=tmp_path / "prepared", runs_root=tmp_path / "runs"),
                  data_processing={**cfg.data_processing, "backend": "local"})
    monkeypatch.setattr(settings, "_SETTINGS", cfg)
    prepared_dataset(tmp_path, "input", monkeypatch)
    card_prompts = []
    def card_model(system, prompt):
        import json
        card_prompts.append(json.loads(prompt))
        assert "Prepare these standard annotations" not in prompt
        assert "concepts" not in json.loads(prompt)
        return {"description": "Dataset facts from Profile"}
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", card_model)
    actions = iter([
        ("prepare_data", {"dataset_id": "new", "sources": [{"id": "source", "kind": "annotation", "root": str(tmp_path / "input/source")}]}),
        ("select_source", {"source_id": "source"}), ("inspect_images", {}),
        ("save_image_plan", {"overrides": {}}), ("normalize_images", {}),
        ("save_harmonization", {"plan": {}}), ("finish", {}), ("finish", {})])
    def model(*args):
        tool, arguments = next(actions)
        return {"tool": tool, "arguments": arguments, "reason": "Prepare one dataset"}
    monkeypatch.setattr("medbenchagent.runtime.harness.complete_json", model)
    graph = compile_benchmark_graph(checkpointer=MemorySaver())
    output = graph.invoke({"user_requirement": "Prepare these standard annotations", "prepare_dataset_id": "new"},
                          {"configurable": {"thread_id": "prepare"}, "recursion_limit": 100})
    assert output["status"] == "complete"
    entry = dataset_by_id(load_dataset_catalog(), "new")
    card, access = load_prepared(entry)
    assert card.total_images == 2
    assert Path(access["annotation_path"]).is_relative_to(tmp_path / "prepared/new")
    assert (Path(access["annotation_path"]).parent / "execution_runtime.json").is_file()
    assert len(card_prompts) == 1
    assert card.preparation_metadata.dataset_description is None
    assert (Path(access["annotation_path"]).parent / "dataset_metadata.json").is_file()


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
