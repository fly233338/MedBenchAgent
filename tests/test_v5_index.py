from medbenchagent.data.asset_manifest import digest_path
from medbenchagent.data.dataset import prepare_annotation_dataset
from medbenchagent.data.v5_models import AssetEntry, CaseRecord, Evidence
from medbenchagent.data.query import DatasetQuery


def test_v5_duckdb_index_and_query(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    asset = AssetEntry(asset_id="a1", kind="image", format="png", source_path="image.png", file_digest=digest_path(source))
    case = CaseRecord(case_id="case-1", patient_id="patient-1", evidence=[
        Evidence(id="asset", type="asset", name="view", value="a1"),
        Evidence(id="label", type="value", name="class", value="positive", ref=["asset"]),
    ])
    (tmp_path / "asset_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    profile, access = prepare_annotation_dataset(tmp_path / "cases.jsonl", tmp_path / "cache")
    assert profile["case_count"] == 1
    query = DatasetQuery(access)
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
