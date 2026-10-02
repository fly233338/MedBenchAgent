import pytest

from medbenchagent.data.asset_manifest import digest_path, iter_asset_manifest, stable_id
from medbenchagent.data.case_query import CaseQuery
from medbenchagent.data.v5_models import AssetEntry, CaseRecord, Evidence, validate_case


def test_v5_case_validates_refs_and_asset_links(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    asset = AssetEntry(asset_id="a1", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(image), width=10, height=10)
    case = CaseRecord(case_id="c1", patient_id=None, evidence=[
        Evidence(id="a", type="asset", name="view", value="a1"),
        Evidence(id="r", type="region", name="lesion", value=[0.1, 0.2, 0.5, 0.8], ref=["a"]),
    ])
    assert validate_case(case, {"a1": asset}).case_id == "c1"

    with pytest.raises(ValueError):
        validate_case(case.model_copy(update={"evidence": [case.evidence[0], case.evidence[0]]}), {"a1": asset})


def test_case_query_profiles(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"pixels")
    asset = AssetEntry(asset_id="a1", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(source), width=10, height=10)
    (tmp_path / "asset_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    case = CaseRecord(case_id="c1", patient_id="p1", evidence=[
        Evidence(id="a", type="asset", name="view", value="a1"),
        Evidence(id="t", type="text", name="report", value="clear", ref=["a"]),
    ])
    (tmp_path / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    query = CaseQuery(tmp_path / "cases.jsonl", tmp_path / "asset_manifest.jsonl")
    assert query.case_count() == 1
    assert query.patient_count() == 1
    assert query.asset_profile()["assets_per_case"] == {1: 1}
    assert query.evidence_profile()["report"]["types"] == {"text": 1}


def test_manifest_rejects_changed_asset(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"pixels")
    asset = AssetEntry(asset_id="a1", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(source))
    manifest = tmp_path / "asset_manifest.jsonl"
    manifest.write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        list(iter_asset_manifest(manifest, source_root=tmp_path))


def test_stable_id_is_reproducible():
    assert stable_id("source", "study-1", prefix="case") == stable_id("source", "study-1", prefix="case")
