from compilation_fixtures import registry
import json

from medbenchagent.data.sources import digest_path
from medbenchagent.data.index import prepare_dataset
from medbenchagent.data.models import AssetEntry, ImageEntry, CaseRecord, Evidence
from medbenchagent.data.index import DatasetIndex
from medbenchagent.planning.task_space import CapabilityCategory, TaskScenario


def test_canonical_duckdb_index_and_query(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    asset = ImageEntry(path="a1", source_id="s", source_digest="raw", kind="image", format="png", source_path="image.png", file_digest=digest_path(source), width=2, height=2)
    case = CaseRecord(case_id="case-1", patient_id="patient-1", evidence=[
        Evidence(id="asset", type="asset", name="view", value="a1"),
        Evidence(id="label", type="value", name="class", value="positive", ref=["asset"]),
    ])
    (tmp_path / "image_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    profile, access = prepare_dataset(tmp_path / "cases.jsonl", tmp_path / "image_manifest.jsonl", tmp_path / "cache", dataset_id="d")
    assert profile["case_count"] == 1
    query = DatasetIndex(access)
    assert query.records(["case-1"])["case-1"]["case_id"] == "case-1"
    assert query.values("class")["values"][0]["value"] == "positive"
    assert query.sample()["total"] == 1
    assert query.sample(evidence_name="class")["total"] == 1
    assert query.sample(asset_kind="image")["total"] == 1
    assert profile["evidence"][0]["value_types"] == {"string": 1}


def test_asset_adapter_resolves_source_relative_path(tmp_path):
    from PIL import Image
    from medbenchagent.data.asset_adapters import ImageAdapter
    image_path = tmp_path / "image.png"
    Image.new("L", (2, 2), 3).save(image_path)
    asset = AssetEntry(asset_id="a", kind="image", format="png", source_path="image.png", file_digest="x")
    loaded = ImageAdapter().load(asset, source_root=tmp_path)
    assert loaded.metadata["width"] == 2


def test_canonical_node_context_uses_constraint_registry_instance(tmp_path, monkeypatch):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"evidence": [{"name": "tumor_region", "types": {"region": 1}}]}), encoding="utf-8")
    query = object.__new__(DatasetIndex)
    query.access = {"revision": "1", "profile_path": str(profile_path)}
    monkeypatch.setattr(query, "assets", lambda: [{"asset_id": "image", "kind": "image"}])
    monkeypatch.setattr(query, "iter_cases", lambda: iter([
            {"case_id": "keep", "patient_id": None, "evidence": [
                    {"id": "image", "type": "asset", "name": "image", "value": "image", "ref": []},
                    {"id": "region", "type": "region", "name": "tumor_region", "value": [0.1, 0.2, 0.3, 0.4], "ref": ["image"]}
        ]},
        {"case_id": "drop", "patient_id": None, "evidence": []},
    ]))
    monkeypatch.setattr(query, "facts", lambda: {"relations": {}})
    node = TaskScenario(
        scenario_id="scenario-1", input_structure="single-image", operation="localize",
        clinical_target="finding", capability_category=CapabilityCategory.RECOGNITION,
    )
    result = query.node_context(node, registry=registry())
    assert result["scenario_id"] == "scenario-1"
    assert result["capacity_hint"] == 1
    assert result["representative_cases"][0]["case_id"] == "keep"
    assert result["evidence"] == [{"name": "tumor_region", "types": {"region": 1}}]

