import copy
import json
from pathlib import Path

import pytest

from medbenchagent.data import preparation_services as pipeline
from medbenchagent.agents import data_processing
from medbenchagent.data.contracts import PreparationEvidence, Source
from medbenchagent.data.operators import inspect_image_population
from medbenchagent.data.preparation_evidence import semantic_evidence
from medbenchagent.data.sources import inventory_source
from medbenchagent.runtime.clarifications import new_issue, record_answer, resolve_semantic_answer
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.inspection import evidence, evidence_page


def setup_source(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "images.csv").write_text("id,path,label\n001,a.png,A\n001,a.png,B\n002,b.png,C\n003,c.png,D\n", encoding="utf-8")
    (raw / "README.md").write_text("One image per distinct id. label describes pathology.", encoding="utf-8")
    source = Source(id="one", root=str(raw))
    bundle = tmp_path / "bundle"
    manifest = inventory_source(source, bundle / "source_manifests/one")
    c = {"run_directory": str(tmp_path / "run"), "operation_directory": str(tmp_path / "ops"),
         "bundle_root": str(bundle), "dataset_scope": "dataset", "source_scope": "one",
         "sources": [source.model_dump()], "active_source": source.model_dump(), "source_root": str(raw),
         "source_manifest": str(manifest), "execution_runtime": {"backend": "local", "requested_backend": "local"}}
    publish(c, "Source", {"manifest": str(manifest)}, ())
    def run(context, output, *, request, **kwargs):
        assert request["operation"] == "image_population"
        return inspect_image_population(context["source_root"], source, request["population"], output)
    monkeypatch.setattr(data_processing, "_run", run)
    return c


def call(c, function, **kwargs):
    c["operation_directory"] = str(Path(c["run_directory"]) / f"op-{evidence_page(c)['total']}")
    outcome = function(c, **kwargs)
    c.update(outcome["updates"])
    return json.loads(outcome["summary"])["inspection_reference"]


def population():
    return {"steps": [{"op": "read", "path": "images.csv", "output": "images"}],
            "input": "images", "image_id": "id", "image_path": "path"}


def grounded_spec(c):
    count = call(c, pipeline.inspect_image_population, population=population())
    doc = call(c, data_processing.read_text, path="README.md")
    c.update(pipeline.save_source_fingerprint(c, inspection_references=[count, doc])["updates"])
    observed = {"meaning": "Preserve source values", "inspection_references": [count],
                "interpretation": "preserve"}
    return {"population": {**observed, "meaning": "One image per distinct id",
                           "interpretation": "interpreted",
                           "semantic_evidence": [{"source": "document", "reference": doc,
                                                  "quote": "One image per distinct id"}]},
            "population_reference": count, "expected_images": 3,
            "identities": {entity: observed for entity in ("patient", "case", "study", "image")},
            "fields": [{**observed, "target": target, "source_path": "images.csv", "source_field": column}
                       for target, column in (("image.image_id", "id"), ("image.image_path", "path"))],
            "missing": observed, "findings": observed, "spatial": observed}


def test_population_counts_unique_images_after_join_and_filter(tmp_path):
    (tmp_path / "images.csv").write_text("id,path,label\n001,a.png,A\n001,a.png,B\n002,b.png,C\n003,c.png,D\n", encoding="utf-8")
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
    assert result["rows"] == 2 and result["unique_images"] == 1
    assert not (tmp_path / "out/annotation.jsonl").exists()


@pytest.mark.parametrize("contents", ["id,path\n\"\",a.png\n", "id,path\n"])
def test_population_rejects_missing_or_empty_identities(tmp_path, contents):
    (tmp_path / "images.csv").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty image identities"):
        inspect_image_population(tmp_path, Source(id="one", root=str(tmp_path)), population(), tmp_path / "out")


@pytest.mark.parametrize("fault,match", [
    ("count", "expected_images"), ("reference", "Population evidence must belong"),
    ("semantic", "Source semantics"), ("quote", "exact quotation"),
    ("foreign", "another dataset"), ("stale", "stale"),
])
def test_semantic_spec_rejects_unfounded_population(tmp_path, monkeypatch, fault, match):
    c = setup_source(tmp_path, monkeypatch)
    spec = grounded_spec(c)
    if fault == "count":
        spec["expected_images"] = 4  # Raw row count is not image count.
    elif fault == "reference":
        spec["population_reference"] = "invented"
    elif fault == "semantic":
        spec["population"]["semantic_evidence"] = []
    elif fault == "quote":
        spec["population"]["semantic_evidence"][0]["quote"] = "Invented population"
    elif fault == "foreign":
        item = spec["population"]["semantic_evidence"][0]
        record = evidence(c, item["reference"])
        from medbenchagent.runtime.inspection import save_evidence
        other = {**c, "dataset_scope": "other"}
        item["reference"] = save_evidence(other, record["operation"], record["arguments"],
                                          record["value"], (), source_ids=["one"])
    else:
        publish(c, "Source", {"revision": 2}, ())
    with pytest.raises(ValueError, match=match):
        pipeline.save_semantic_spec(c, spec=spec)
    assert not (Path(c["operation_directory"]) / "spec.json").exists()


def test_semantic_spec_receipts_survive_private_state_reload(tmp_path, monkeypatch):
    c = setup_source(tmp_path, monkeypatch)
    spec = grounded_spec(c)
    restored = json.loads(json.dumps(c))
    index = pipeline.describe(restored)["inspection_references"]["entries"]
    assert any(row["reference"] == spec["population_reference"] and row["operation"] == "image_population" for row in index)
    assert all("value" not in row for row in index)
    result = pipeline.save_semantic_spec(restored, spec=spec)
    saved = json.loads(Path(result["updates"]["semantic_spec_path"]).read_text(encoding="utf-8"))
    assert saved["expected_images"] == 3
    assert saved["population_reference"] == spec["population_reference"]
    publish(restored, "Source", {"revision": 2}, ())
    assert pipeline.describe(restored)["inspection_references"]["entries"] == []


def test_user_semantic_evidence_is_scoped_and_versioned(tmp_path, monkeypatch):
    c = setup_source(tmp_path, monkeypatch)
    issue = new_issue("DataProcessingAgent", c, "population", "Which images?")
    issue = resolve_semantic_answer(c, record_answer(c, issue, "Use each distinct image id"))
    c["issues"] = {issue["issue_id"]: issue}
    evidence = [PreparationEvidence(source="user", reference=issue["answer_reference"], quote="each distinct image id")]
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
                 "target": "image.annotation.label", "reason": "Source definition",
                 "inspection_references": [field]}
    with pytest.raises(ValueError, match="Source semantics"):
        pipeline.save_harmonization(c, plan={"fields": [alignment]})
    alignment["semantic_evidence"] = [{"source": "document", "reference": doc, "quote": "label describes pathology"}]
    assert pipeline.save_harmonization(c, plan={"fields": [alignment]})["status"] == "success"
    separate = {"source_id": "one", "field": "image.annotation.label", "action": "keep_separate",
                "reason": "Keep original meaning", "inspection_references": [field]}
    assert pipeline.save_harmonization(c, plan={"fields": [separate]})["status"] == "success"
    assert pipeline.save_harmonization(c, plan={})["status"] == "success"
    altered = {**alignment, "field": "image.annotation.nonexistent"}
    with pytest.raises(ValueError, match="actual normalized source field"):
        pipeline.save_harmonization(c, plan={"fields": [altered]})
    publish(c, "NormalizedSource", {"revision": 2}, ("Source",))
    with pytest.raises(ValueError, match="stale"):
        pipeline.save_harmonization(c, plan={"fields": [alignment]})
