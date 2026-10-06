import json
from pathlib import Path

import pytest

import medbenchagent.preparation.validation as preparation_validation
from medbenchagent.data.asset_manifest import digest_path
from medbenchagent.data.models import AssetEntry, CaseRecord, Evidence
from medbenchagent.data.preparation_validation import validate_candidate
from medbenchagent.preparation.contracts import (
    CaseIdentityRule, FieldRule, GroupingRule, PopulationPolicy, PopulationRecord,
    PopulationSummary, PreparationMapping, QuestionSet, SemanticQuestion, SourceReport,
    Stage, Submission, ValidationIssue,
)
from medbenchagent.preparation.validation import (
    _semantic_control, build_population_summary, validate_handoff,
    validate_understand_candidate, write_population_projection,
)


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


def test_only_explicit_semantic_issue_codes_request_mapping_revision():
    semantic = ValidationIssue(
        stage=Stage.PREPARE, issue_id="semantic", code="grouping_semantics_conflict",
        message="accepted grouping semantics conflict with source",
    )
    implementation = ValidationIssue(
        stage=Stage.PREPARE, issue_id="manifest", code="output_manifest_failure",
        message="output_manifest mapping digest does not match accepted mapping",
    )

    assert _semantic_control([semantic]) == "mapping_revision_required"
    assert _semantic_control([implementation]) is None


def test_output_manifest_mapping_failure_stays_in_prepare(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in (
        "scripts/prepare.py", "cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl",
        "conversion_config.json", "grouping_policy.json", "execution_report.json",
        "output_manifest.json",
    ):
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (candidate / "submission.json").write_text(
        json.dumps({"requested_outcome": "submitted"}), encoding="utf-8",
    )
    monkeypatch.setattr(preparation_validation, "validate_candidate", lambda *args, **kwargs: {
        "failures": [{"kind": "output_manifest_failure",
                       "error": "output_manifest mapping digest does not match accepted mapping"}],
        "accepted_cases": 1, "systemic_failure": False,
        "population": {"expected_units": 1, "accounted_units": 1, "unprocessed_units": 0},
    })

    result = preparation_validation.validate_prepare_candidate(
        candidate, state={"preparation_root": str(tmp_path / "preparation")},
    )

    assert result.control is None
    assert result.status == "rejected"


def _understand_candidate(tmp_path, *, report_count=2):
    candidate = tmp_path / "understand"
    candidate.mkdir()
    records = [
        PopulationRecord(source_id="source", source_unit_id="unit-1", locator="one.nii.gz", unit_type="volume_case"),
        PopulationRecord(source_id="source", source_unit_id="unit-2", locator="two.nii.gz", unit_type="volume_case"),
    ]
    report = SourceReport(
        sources=[{"source_id": "source"}], tables=[], documents=[], checks=[{"name": "inventory"}],
        relationships=[], population=PopulationSummary(
            unit_count=report_count, unit_type="volume_case", source_counts={"source": report_count},
        ), limitations=[],
    )
    mapping = PreparationMapping(
        population_policy=PopulationPolicy(
            rule_id="population", unit_type="volume_case", source_ids=["source"],
            identity_fields=["id"], inclusion="all", evidence_refs=["source:inventory"],
        ),
        case_identity=CaseIdentityRule(
            rule_id="case", target_path="case_id", fields=["id"], evidence_refs=["source:inventory"],
        ),
        grouping=GroupingRule(
            rule_id="group", case_unit="volume_case", key_fields=["id"], evidence_refs=["source:inventory"],
        ),
        fields=[FieldRule(
            rule_id="field", source_path="id", disposition="grouping_metadata", source_type="string",
            target_type="string", missingness="reject", reason="identity", scope="unit",
            evidence_refs=["source:inventory"],
        )],
    )
    (candidate / "source_report.json").write_text(report.model_dump_json(), encoding="utf-8")
    (candidate / "mapping.json").write_text(mapping.model_dump_json(), encoding="utf-8")
    (candidate / "image_population.jsonl").write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8",
    )
    (candidate / "semantic_questions.json").write_text(QuestionSet(questions=[]).model_dump_json(), encoding="utf-8")
    (candidate / "inspect_source.py").write_text("print('ok')\n", encoding="utf-8")
    (candidate / "submission.json").write_text(Submission(requested_outcome="submitted").model_dump_json(), encoding="utf-8")
    return candidate


def test_population_summary_and_projection_share_one_source(tmp_path):
    records = [
        PopulationRecord(source_id="a", source_unit_id="1", locator="1", unit_type="case", label_status="labeled"),
        PopulationRecord(source_id="a", source_unit_id="2", locator="2", unit_type="case", label_status="unlabeled"),
    ]
    summary = build_population_summary(records)
    assert summary.unit_count == 2
    assert summary.source_counts == {"a": 2}
    report = write_population_projection(records, report={"sources": [{"source_id": "a"}], "checks": [{}]}, output_dir=tmp_path)
    assert report["population"] == summary.model_dump(mode="json")
    assert len((tmp_path / "image_population.jsonl").read_text(encoding="utf-8").splitlines()) == summary.unit_count


def test_understand_validator_reports_structured_population_count_mismatch(tmp_path):
    candidate = _understand_candidate(tmp_path, report_count=1)
    result = validate_understand_candidate(
        {path.name: str(path) for path in candidate.iterdir()}, known_source_ids=["source"],
    )
    issue = next(item for item in result.issues if item.code == "population_count_mismatch")
    assert result.status == "rejected"
    assert issue.expected == "image_population rows=2"
    assert issue.observed == "report.population.unit_count=1"
    assert issue.details == {"reported_unit_count": 1, "population_rows": 2}


def test_source_report_requires_typed_population_summary():
    with pytest.raises(ValueError):
        SourceReport.model_validate({
            "sources": [{"source_id": "source"}], "checks": [{}], "population": {},
        })


def test_host_and_worker_understand_validation_share_issue_code(tmp_path):
    candidate = _understand_candidate(tmp_path, report_count=1)
    outputs = {path.name: str(path) for path in candidate.iterdir()}
    worker_result = validate_understand_candidate(outputs, known_source_ids=["source"])
    host_result = validate_handoff(outputs, state={"source_roots": {"source": str(tmp_path)}}, replay=False)
    assert [item.code for item in worker_result.issues] == [item.code for item in host_result.issues]


def test_understand_validator_preserves_blocking_questions_as_needs_input(tmp_path):
    candidate = _understand_candidate(tmp_path)
    question = SemanticQuestion(
        issue_id="meaning", stage=Stage.UNDERSTAND, question="What does this field mean?",
        impact="changes grouping", blocking=True, resolved=False,
    )
    (candidate / "semantic_questions.json").write_text(
        QuestionSet(questions=[question]).model_dump_json(), encoding="utf-8",
    )
    result = validate_understand_candidate(
        {path.name: str(path) for path in candidate.iterdir()}, known_source_ids=["source"],
    )
    assert result.status == "needs_input"
    assert result.questions[0].issue_id == "meaning"
