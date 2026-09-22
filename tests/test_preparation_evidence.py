import copy
import json
from pathlib import Path

import pytest

from medbenchagent.agents import data_pipeline as pipeline, data_processing
from medbenchagent.data.contracts import PreparationEvidence, Source
from medbenchagent.data.operators import inspect_image_population
from medbenchagent.data.preparation_evidence import semantic_evidence
from medbenchagent.data.sources import inventory_source
from medbenchagent.runtime.clarifications import new_issue
from medbenchagent.runtime.dependencies import publish


def setup_source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "images.csv").write_text("id,label\n001,A\n001,B\n002,C\n003,D\n", encoding="utf-8")
    (raw / "README.md").write_text("One image per distinct id. label describes pathology.", encoding="utf-8")
    source = Source(id="one", root=str(raw))
    bundle = tmp_path / "bundle"
    manifest = inventory_source(source, bundle / "source_manifests/one")
    c = {"run_directory": str(tmp_path / "run"), "operation_directory": str(tmp_path / "ops"),
         "bundle_root": str(bundle), "dataset_scope": "dataset", "source_scope": "one",
         "sources": [source.model_dump()], "active_source": source.model_dump(), "source_root": str(raw),
         "source_manifest": str(manifest)}
    publish(c, "Source", {"manifest": str(manifest)}, ())
    def run(context, output, *, request, **kwargs):
        assert request["operation"] == "image_population"
        return inspect_image_population(context["source_root"], source, request["population"], output)
    monkeypatch.setattr(data_processing, "_run", run)
    return c


def call(c, function, **kwargs):
    c["operation_directory"] = str(Path(c["run_directory"]) / f"op-{len(c.get('inspections', {}))}")
    outcome = function(c, **kwargs)
    c.update(outcome["updates"])
    return json.loads(outcome["summary"])["inspection_reference"]


def population():
    return {"steps": [{"op": "read", "path": "images.csv", "output": "images"}],
            "input": "images", "image_id": "id"}


def grounded_plan(c):
    count = call(c, pipeline.inspect_image_population, population=population())
    doc = call(c, data_processing.read_text, path="README.md")
    return {"mapping": {"authoritative_population": "All distinct image IDs", "expected_images": 3,
                        "entity_meanings": {}, "evidence": ["Source codebook"],
                        "population_reference": count, "inspection_references": [count],
                        "semantic_evidence": [{"source": "document", "reference": doc,
                                               "quote": "One image per distinct id"}]},
            "fallback_reason": "Custom label aggregation"}


def test_population_counts_unique_images_after_join_and_filter(tmp_path):
    (tmp_path / "images.csv").write_text("id,label\n001,A\n001,B\n002,C\n003,D\n", encoding="utf-8")
    (tmp_path / "cohort.csv").write_text("id,cohort\n001,yes\n002,no\n003,no\n", encoding="utf-8")
    spec = population()
    spec["steps"] += [
        {"op": "read", "path": "cohort.csv", "output": "cohort"},
        {"op": "join", "input": "images", "right": "cohort", "output": "joined",
         "left_keys": ["id"], "right_keys": ["id"], "cardinality": "many_to_one"},
        {"op": "filter", "input": "joined", "output": "eligible", "field": "cohort",
         "values": ["yes"], "reason": "Documented cohort"},
    ]
    spec["input"] = "eligible"
    result = inspect_image_population(tmp_path, Source(id="one", root=str(tmp_path)), spec, tmp_path / "out")
    assert result == {"rows": 2, "unique_images": 1, "image_id": "id"}
    assert not (tmp_path / "out/annotation.jsonl").exists()


@pytest.mark.parametrize("contents", ["id\n\"\"\n", "id\n"])
def test_population_rejects_missing_or_empty_identities(tmp_path, contents):
    (tmp_path / "images.csv").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty image identities"):
        inspect_image_population(tmp_path, Source(id="one", root=str(tmp_path)), population(), tmp_path / "out")


@pytest.mark.parametrize("fault,match", [
    ("count", "expected_images"), ("reference", "Inspection reference"),
    ("semantic", "Source semantics"), ("quote", "exact quotation"),
    ("foreign", "another dataset/source"), ("stale", "stale"),
])
def test_mapping_rejects_unfounded_population(tmp_path, monkeypatch, fault, match):
    c = setup_source(tmp_path, monkeypatch)
    plan = grounded_plan(c)
    if fault == "count":
        plan["mapping"]["expected_images"] = 4  # Raw row count is not image count.
    elif fault == "reference":
        plan["mapping"]["population_reference"] = "invented"
    elif fault == "semantic":
        plan["mapping"]["semantic_evidence"] = []
    elif fault == "quote":
        plan["mapping"]["semantic_evidence"][0]["quote"] = "Invented population"
    elif fault == "foreign":
        c["inspections"][plan["mapping"]["population_reference"]]["dataset_id"] = "other"
    else:
        publish(c, "Source", {"revision": 2}, ())
    with pytest.raises(ValueError, match=match):
        pipeline.save_preparation_plan(c, plan=plan)
    assert not (Path(c["operation_directory"]) / "preparation.json").exists()


def test_mapping_receipts_survive_private_state_reload(tmp_path, monkeypatch):
    c = setup_source(tmp_path, monkeypatch)
    plan = grounded_plan(c)
    restored = json.loads(json.dumps(c))
    index = pipeline.describe(restored)["inspection_references"]
    assert index[plan["mapping"]["population_reference"]]["operation"] == "image_population"
    assert "value" not in index[plan["mapping"]["population_reference"]]
    result = pipeline.save_preparation_plan(restored, plan=plan)
    saved = json.loads(Path(result["updates"]["preparation_plan_path"]).read_text(encoding="utf-8"))
    assert saved["mapping"]["expected_images"] == 3
    assert saved["mapping"]["population_reference"] == plan["mapping"]["population_reference"]
    publish(restored, "Source", {"revision": 2}, ())
    assert pipeline.describe(restored)["inspection_references"] == {}


def test_user_semantic_evidence_is_scoped_and_versioned(tmp_path, monkeypatch):
    c = setup_source(tmp_path, monkeypatch)
    issue = new_issue("DataProcessingAgent", c, "population", "Which images?")
    issue.update(status="resolved", kind="semantic", answer="Use each distinct image id")
    c["issues"] = {issue["issue_id"]: issue}
    evidence = [PreparationEvidence(source="user", reference=issue["issue_id"], quote="each distinct image id")]
    semantic_evidence(c, evidence, "one")
    foreign = copy.deepcopy(c)
    foreign["issues"][issue["issue_id"]]["dataset_id"] = "another"
    with pytest.raises(ValueError, match="resolved decision"):
        semantic_evidence(foreign, evidence, "one")
    publish(c, "Source", {"revision": 2}, ())
    with pytest.raises(ValueError, match="stale"):
        semantic_evidence(c, evidence, "one")


def test_harmonization_requires_current_field_and_semantic_evidence(tmp_path, monkeypatch):
    from test_data_layer import record
    c = setup_source(tmp_path, monkeypatch)
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    (normalized / "annotation.jsonl").write_text(json.dumps(record()) + "\n", encoding="utf-8")
    c["normalized_sources"] = {"one": str(normalized)}
    publish(c, "NormalizedSource", {"path": str(normalized)}, ("Source",))
    field = call(c, pipeline.inspect_normalized, source_id="one", field="image.annotation.label")
    doc = call(c, data_processing.read_text, source_id="one", path="README.md")
    alignment = {"source_id": "one", "field": "image.annotation.label", "action": "merge",
                 "target": "image.annotation.label", "evidence": "Source definition",
                 "inspection_references": [field]}
    with pytest.raises(ValueError, match="Source semantics"):
        pipeline.save_harmonization(c, plan={"fields": [alignment]})
    alignment["semantic_evidence"] = [{"source": "document", "reference": doc, "quote": "label describes pathology"}]
    assert pipeline.save_harmonization(c, plan={"fields": [alignment]})["status"] == "success"
    separate = {"source_id": "one", "field": "image.annotation.label", "action": "keep_separate",
                "evidence": "Keep original meaning", "inspection_references": [field]}
    assert pipeline.save_harmonization(c, plan={"fields": [separate]})["status"] == "success"
    assert pipeline.save_harmonization(c, plan={})["status"] == "success"
    altered = {**alignment, "field": "image.annotation.nonexistent"}
    with pytest.raises(ValueError, match="actual normalized source field"):
        pipeline.save_harmonization(c, plan={"fields": [altered]})
    publish(c, "NormalizedSource", {"revision": 2}, ("Source",))
    with pytest.raises(ValueError, match="stale"):
        pipeline.save_harmonization(c, plan={"fields": [alignment]})
