import json
from pathlib import Path
import pytest
from medbenchagent.agents.data_pipeline import save_dataset_metadata
from medbenchagent.data.contracts import Source
from medbenchagent.data.dataset_card import PreparationMetadata
from medbenchagent.data.sources import inventory_source
from medbenchagent.runtime.dependencies import publish


def test_metadata_is_extracted_from_verified_quotes_not_benchmark_goal(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    quote = "Example Dataset contains ultrasound images and expert annotations."
    (root / "README.md").write_text(quote, encoding="utf-8")
    source = Source(id="s", root=str(root))
    bundle = tmp_path / "bundle"
    inventory_source(source, bundle / "source_manifests/s")
    context = {"run_directory": str(tmp_path / "run"), "dataset_scope": "a",
               "bundle_root": str(bundle), "sources": [source.model_dump()],
               "user_requirement": "GOAL_LOCALIZATION_ONLY"}
    publish(context, "source/s/Source", {}, ())
    prompts = []
    def model(system, prompt):
        prompts.append(prompt)
        assert "GOAL_" not in prompt
        assert json.loads(prompt)["source_references"][0]["quote"] == quote
        return {"dataset_name": "Example Dataset", "dataset_description": "Ultrasound images",
                "source_description": "Expert annotations"}
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", model)
    refs = [{"source_id": "s", "path": "README.md", "quote": quote}]
    output = save_dataset_metadata(context, source_references=refs)
    path = Path(output["updates"]["dataset_metadata_path"])
    first = path.read_bytes()
    context["user_requirement"] = "GOAL_DIAGNOSIS_ONLY"
    save_dataset_metadata(context, source_references=refs)
    assert prompts[0] == prompts[1] and path.read_bytes() == first
    assert json.loads(first)["source_references"] == [{**r, "page": 1} for r in refs]
    with pytest.raises(ValueError, match="verbatim"):
        save_dataset_metadata(context, source_references=[{**refs[0], "quote": "Invented suitability"}])
    assert len(prompts) == 2 and path.read_bytes() == first


def test_unreferenced_preparation_description_is_rejected():
    with pytest.raises(ValueError, match="source references"):
        PreparationMetadata(dataset_description="Unsupported clinical purpose")


def test_finish_requires_explicit_metadata_decision(tmp_path, monkeypatch):
    from medbenchagent.agents.data_pipeline import finish, available_tools
    c = {"run_directory": str(tmp_path / "run"), "bundle_root": str(tmp_path / "bundle"),
         "sources": [{"id": "s"}], "normalized_sources": {"s": "normalized"}}
    publish(c, "source/s/NormalizedSource", {}, ())
    publish(c, "Harmonization", {}, ())
    assert "finish" not in available_tools(c)
    with pytest.raises(ValueError, match="Record dataset metadata first"):
        finish(c)
    assert not (tmp_path / "bundle/ready.json").exists()
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", lambda *args: pytest.fail("Empty metadata needs no model"))
    c.update(save_dataset_metadata(c, source_references=[])["updates"])
    assert "finish" in available_tools(c)
    metadata = PreparationMetadata.model_validate_json(Path(c["dataset_metadata_path"]).read_text(encoding="utf-8"))
    assert metadata.source_references == [] and metadata.dataset_name is None
