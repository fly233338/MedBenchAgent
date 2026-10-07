import pytest

from medbenchagent.data.sources import digest_path, stable_id
from medbenchagent.data.manifests import iter_image_manifest
from medbenchagent.data.index import prepare_dataset, DatasetIndex
from medbenchagent.data.models import ImageEntry, CaseRecord, Evidence, validate_case


def test_canonical_case_validates_refs_and_asset_links(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    asset = ImageEntry(path="a1", source_id="s", source_digest="original", kind="image", format="png", source_path="image.png",
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
    asset = ImageEntry(path="a1", source_id="s", source_digest="original", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(source), width=10, height=10)
    (tmp_path / "image_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    case = CaseRecord(case_id="c1", patient_id="p1", evidence=[
        Evidence(id="a", type="asset", name="view", value="a1"),
        Evidence(id="t", type="text", name="report", value="clear", ref=["a"]),
    ])
    (tmp_path / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    profile, access = prepare_dataset(tmp_path / "cases.jsonl", tmp_path / "image_manifest.jsonl", tmp_path / "cache")
    query = DatasetIndex(access)
    assert query.sample()["total"] == 1
    assert profile["patient_count"] == 1
    assert profile["assets"]["assets_per_case"] == {1: 1}
    assert next(row for row in profile["evidence"] if row["name"] == "report")["types"] == {"text": 1}


def test_manifest_rejects_duplicate_path(tmp_path):
    image = ImageEntry(path="images/a.png", source_id="s", source_path="raw.png", source_digest="raw",
                       file_digest="output", format="png", width=10, height=10)
    manifest = tmp_path / "image_manifest.jsonl"
    manifest.write_text((image.model_dump_json() + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        list(iter_image_manifest(manifest))


def test_stable_id_is_reproducible():
    assert stable_id("source", "study-1", prefix="case") == stable_id("source", "study-1", prefix="case")

