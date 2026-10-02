from medbenchagent.data.asset_manifest import digest_path
from medbenchagent.data.finalization import finalize_v5_prepared_dataset
from medbenchagent.data.v5_models import AssetEntry, CaseRecord, Evidence
from medbenchagent.data.v5_catalog import verify_v5_bundle
from medbenchagent.data.dataset_catalog import prepared_bundle_entry, register_dataset, load_prepared
from medbenchagent.data.contracts import Source
from medbenchagent.runtime import preparation_context


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
    (candidate / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "provenance.jsonl").write_text('{"source_id":"s1","record_ref":"record:1"}\n', encoding="utf-8")
    (candidate / "conversion_config.json").write_text('{}', encoding="utf-8")
    (candidate / "grouping_policy.json").write_text('{}', encoding="utf-8")
    (candidate / "mapping.json").write_text('{}', encoding="utf-8")
    (candidate / "conversion_version").write_text("v5\n", encoding="utf-8")
    (candidate / "complete.json").write_text('{"complete":true}', encoding="utf-8")
    result = finalize_v5_prepared_dataset(candidate, {"bundle_root": str(tmp_path / "bundle")})
    assert result["status"] == "complete"
    assert (tmp_path / "bundle" / "ready.json").is_file()
    assert result["report"]["accepted_cases"] == 1
    assert verify_v5_bundle(tmp_path / "bundle")["card"]["schema_version"] == "dataset-card.v3"


def test_direct_prepared_bundle_is_catalog_reusable(tmp_path):
    prepared = tmp_path / "Prepared" / "breast"
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    asset_path = candidate / "image.png"
    asset_path.write_bytes(b"image")
    asset = AssetEntry(asset_id="a1", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(asset_path), width=4, height=4)
    case = CaseRecord(case_id="c1", patient_id=None,
                      evidence=[Evidence(id="a", type="asset", name="image", value="a1")])
    (candidate / "asset_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "provenance.jsonl").write_text('{"source_id":"s1","record_ref":"record:1"}\n', encoding="utf-8")
    for name, value in (("conversion_config.json", "{}"), ("grouping_policy.json", "{}"),
                        ("mapping.json", "{}"), ("complete.json", '{"complete":true}')):
        (candidate / name).write_text(value, encoding="utf-8")
    (candidate / "conversion_version").write_text("v5\n", encoding="utf-8")

    result = finalize_v5_prepared_dataset(candidate, {"bundle_root": str(prepared), "dataset_id": "breast"})
    entry = prepared_bundle_entry("breast", prepared_root=tmp_path / "Prepared")
    catalog = tmp_path / "catalog.json"
    registered = register_dataset("breast", prepared, catalog_path=catalog)
    card, access = load_prepared(registered)

    assert result["bundle"] == str(prepared.resolve())
    assert entry["cases_path"] == str(prepared / "cases.jsonl")
    assert registered["cases_path"] == entry["cases_path"]
    assert card["dataset_id"] == "breast"
    assert access["cases_path"] == entry["cases_path"]


def test_preparation_context_uses_direct_dataset_bundle(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "source.csv").write_text("id\n1\n", encoding="utf-8")
    prepared = tmp_path / "Prepared"
    settings = type("Settings", (), {
        "paths": type("Paths", (), {"prepared_datasets_root": prepared})(),
        "data_processing": {"coding_worker_max_iterations": 1, "coding_worker_timeout_seconds": 1},
    })()
    monkeypatch.setattr(preparation_context, "get_settings", lambda: settings)
    monkeypatch.setattr(preparation_context, "data_dependency_identity", lambda: {"test": "1"})
    monkeypatch.setattr(preparation_context, "publish", lambda *_args, **_kwargs: {})
    context = {
        "run_id": "run",
        "operation_directory": str(tmp_path / "operation"),
        "execution_runtime": {"backend": "local", "requested_backend": "local"},
    }

    result = preparation_context.initialize_preparation(
        context, [Source(id="original", root=str(raw))], "breast"
    )

    assert result["bundle_root"] == str((prepared / "breast").resolve())
    assert (prepared / "breast" / "source_manifests" / "original" / "source.json").is_file()
