from pathlib import Path
import json

from medbenchagent.data.asset_manifest import digest_path
from medbenchagent.data.preparation_validation import validate_candidate
from medbenchagent.data.models import AssetEntry, CaseRecord, Evidence
from medbenchagent.data.bundle_catalog import verify_bundle
from medbenchagent.data.dataset_catalog import prepared_bundle_entry, register_dataset, load_prepared
from medbenchagent.data.contracts import Source
from medbenchagent.preparation import bootstrap as preparation_context
from medbenchagent.preparation import DatasetPublisher, Stage, ValidationResult
from medbenchagent.data import readiness


def publish_candidate(candidate, context):
    report = validate_candidate(candidate)
    ledger = DatasetPublisher._build_ledger(candidate)
    validation = ValidationResult(
        status="accepted", stage=Stage.PREPARE, report=report,
        candidate_digest=DatasetPublisher._candidate_digest(candidate, ledger),
        digest_ledger=ledger,
    )
    return DatasetPublisher(context["bundle_root"]).publish(candidate, context, validation)


def test_candidate_digest_ledger_reuses_unchanged_files(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    payload = candidate / "candidate.json"
    payload.write_text("candidate", encoding="utf-8")
    ledger = DatasetPublisher._build_ledger(candidate)

    def fail_digest(_path):
        raise AssertionError("unchanged candidate file should not be read again")

    monkeypatch.setattr("medbenchagent.preparation.publisher.digest_file", fail_digest)

    reused = DatasetPublisher._build_ledger(candidate, ledger)

    assert reused["candidate.json"].digest == ledger["candidate.json"].digest


def test_finalize_canonical_bundle(tmp_path):
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
    (candidate / "provenance.jsonl").write_text('{"source_id":"s1","source_unit_id":"unit-1","record_refs":["record:1"],"case_ids":["c1"],"status":"retained","reason":"mapped"}\n', encoding="utf-8")
    (candidate / "conversion_config.json").write_text('{}', encoding="utf-8")
    (candidate / "grouping_policy.json").write_text('{}', encoding="utf-8")
    (candidate / "mapping.json").write_text('{}', encoding="utf-8")
    (candidate / "complete.json").write_text('{"complete":true}', encoding="utf-8")
    result = publish_candidate(candidate, {"bundle_root": str(tmp_path / "bundle")})
    assert result["status"] == "complete"
    assert (tmp_path / "bundle" / "ready.json").is_file()
    assert result["report"]["accepted_cases"] == 1
    assert verify_bundle(tmp_path / "bundle")["card"]["schema_version"] == "dataset-card"


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
    (candidate / "provenance.jsonl").write_text('{"source_id":"s1","source_unit_id":"unit-1","record_refs":["record:1"],"case_ids":["c1"],"status":"retained","reason":"mapped"}\n', encoding="utf-8")
    for name, value in (("conversion_config.json", "{}"), ("grouping_policy.json", "{}"),
                        ("mapping.json", "{}"), ("complete.json", '{"complete":true}')):
        (candidate / name).write_text(value, encoding="utf-8")

    result = publish_candidate(candidate, {"bundle_root": str(prepared), "dataset_id": "breast"})
    entry = prepared_bundle_entry("breast", prepared_root=tmp_path / "Prepared")
    catalog = tmp_path / "catalog.json"
    registered = register_dataset("breast", prepared, catalog_path=catalog)
    assert list(json.loads(catalog.read_text(encoding="utf-8"))) == ["datasets"]
    card, access = load_prepared(registered)

    assert result["bundle"] == str(prepared.resolve())
    assert Path(entry["cases_path"]).is_relative_to(prepared / "revisions")
    assert registered["cases_path"] == entry["cases_path"]
    assert card["dataset_id"] == "breast"
    assert access["cases_path"] == entry["cases_path"]
    (prepared / "cases.jsonl").write_text("corrupted projection", encoding="utf-8")
    assert verify_bundle(prepared)["ready"]["access"]["cases_path"] == entry["cases_path"]
    assert prepared_bundle_entry("breast", prepared_root=tmp_path / "Prepared")["cases_path"] == entry["cases_path"]


def test_prepared_directory_inventory_returns_valid_bundles_and_diagnostics(tmp_path, monkeypatch):
    prepared = tmp_path / "Prepared"
    source = tmp_path / "source"
    source.mkdir()
    (source / "image.png").write_bytes(b"image")
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
    (candidate / "provenance.jsonl").write_text('{"source_id":"s1","source_unit_id":"unit-1","record_refs":["record:1"],"case_ids":["c1"],"status":"retained","reason":"mapped"}\n', encoding="utf-8")
    for name, value in (("conversion_config.json", "{}"), ("grouping_policy.json", "{}"),
                        ("mapping.json", "{}"), ("complete.json", '{"complete":true}')):
        (candidate / name).write_text(value, encoding="utf-8")
    publish_candidate(candidate, {"bundle_root": str(prepared / "breast"), "dataset_id": "breast"})
    (prepared / "broken").mkdir(parents=True)
    monkeypatch.setattr(readiness, "get_settings", lambda: type("Settings", (), {
        "paths": type("Paths", (), {"prepared_datasets_root": prepared})(),
        "dataset_sources": [],
    })())

    entries, errors = readiness.prepared_directory_entries({})

    assert [entry["dataset_id"] for entry in entries] == ["breast"]
    assert "broken" in errors


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

