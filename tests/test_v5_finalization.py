from medbenchagent.data.asset_manifest import digest_path
from medbenchagent.data.finalization import finalize_v5_prepared_dataset
from medbenchagent.data.v5_models import AssetEntry, CaseRecord, Evidence


def test_finalize_v5_bundle(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    asset_path = candidate / "image.png"
    asset_path.write_bytes(b"image")
    asset = AssetEntry(asset_id="a1", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(asset_path), width=4, height=4)
    case = CaseRecord(case_id="c1", patient_id=None,
                      evidence=[Evidence(id="a", type="asset", name="image", value="a1")])
    (candidate / "asset_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "annotation.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "provenance.jsonl").write_text('{"source_id":"s1","record_ref":"record:1"}\n', encoding="utf-8")
    result = finalize_v5_prepared_dataset(candidate, {"bundle_root": str(tmp_path / "bundle")})
    assert result["status"] == "complete"
    assert (tmp_path / "bundle" / "ready.json").is_file()
    assert result["report"]["accepted_cases"] == 1
