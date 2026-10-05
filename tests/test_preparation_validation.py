import json
from pathlib import Path

from medbenchagent.data.asset_manifest import digest_path
from medbenchagent.data.models import AssetEntry, CaseRecord, Evidence
from medbenchagent.data.preparation_validation import validate_candidate


def _candidate(tmp_path: Path, *, record_ref="unit-1", dependency_path=None):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    image = candidate / "image.png"
    image.write_bytes(b"image")
    asset = AssetEntry(asset_id="asset-1", kind="image", format="png", source_path="image.png",
                       file_digest=digest_path(image), width=4, height=4)
    case = CaseRecord(case_id="case-1", patient_id=None,
                      evidence=[Evidence(id="image", type="asset", name="image", value="asset-1")])
    (candidate / "asset_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "cases.jsonl").write_text(case.model_dump_json() + "\n", encoding="utf-8")
    (candidate / "provenance.jsonl").write_text(
        json.dumps({"source_id": "source", "record_ref": record_ref, "source_unit_id": record_ref,
                    "case_id": "case-1"}) + "\n", encoding="utf-8")
    for name in ("conversion_config.json", "grouping_policy.json", "execution_report.json", "mapping.json"):
        (candidate / name).write_text("{}", encoding="utf-8")
    population = tmp_path / "image_population.jsonl"
    population.write_text('{"unit_id":"unit-1"}\n', encoding="utf-8")
    entries = []
    for name in ("cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl", "conversion_config.json",
                 "grouping_policy.json", "execution_report.json"):
        path = candidate / name
        entries.append({"path": name, "kind": "output", "digest": digest_path(path)})
    if dependency_path is not None:
        dependencies = [{"path": dependency_path, "role": "helper"}]
    else:
        dependencies = []
    (candidate / "output_manifest.json").write_text(json.dumps({"files": entries, "dependencies": dependencies}), encoding="utf-8")
    (candidate / "submission.json").write_text(json.dumps({"requested_outcome": "submitted"}), encoding="utf-8")
    return candidate, population


def test_population_reconciliation_reports_missing_source_unit(tmp_path):
    candidate, population = _candidate(tmp_path, record_ref="other-unit")

    report = validate_candidate(candidate, population_path=population)

    assert report["systemic_failure"] is True
    assert any(item["kind"] == "population_reconciliation_failure" for item in report["failures"])


def test_output_manifest_dependency_must_exist_inside_candidate(tmp_path):
    candidate, _ = _candidate(tmp_path, dependency_path="missing-helper.py")

    report = validate_candidate(candidate)

    assert report["systemic_failure"] is True
    assert any(item["kind"] == "output_manifest_failure" for item in report["failures"])
