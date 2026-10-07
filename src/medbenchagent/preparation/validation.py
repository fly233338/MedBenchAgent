"""Two host gates: accept Understand semantics, then validate Prepare data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from medbenchagent.data.preparation_validation import validate_candidate
from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import (
    CONTROL_MAPPING_REVISION_REQUIRED,
    SEMANTIC_REVISION_CODES,
    AcceptedHandoff, ArtifactRef, DigestEntry, PopulationRecord, PopulationSummary,
    PreparationMapping, QuestionSet, Stage, SourceReport,
    Submission, ImageConversionPlan, ValidationIssue, ValidationResult, STAGE_SPECS, UserDecision,
    digest_file, digest_payload,
)


def _issue(stage, code, message, **values):
    return ValidationIssue(stage=stage, code=code, message=message,
                           issue_id=f"{stage.value}:{code}", **values)


# The acceptance surface is kept next to the validators so adding a host check
# without making its worker visibility explicit is easy to detect in review.
PREPARATION_CONTRACT_AUDIT = (
    {"stage": "understand", "code": "missing_output", "inputs": ["required outputs"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "source_report_schema", "inputs": ["source_report.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "source_report_incomplete", "inputs": ["source_report.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "mapping_schema", "inputs": ["mapping.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "mapping_incomplete", "inputs": ["mapping.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "population_schema", "inputs": ["image_population.jsonl"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "population_empty", "inputs": ["image_population.jsonl"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "population_duplicate_identity", "inputs": ["image_population.jsonl"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "population_unknown_source", "inputs": ["image_population.jsonl", "source_index.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "population_count_mismatch", "inputs": ["source_report.json", "image_population.jsonl"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "question_schema", "inputs": ["semantic_questions.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "question_stage_mismatch", "inputs": ["semantic_questions.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "submission_schema", "inputs": ["submission.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "submission_outcome", "inputs": ["submission.json"], "worker_visible": True, "host_only": False},
    {"stage": "understand", "code": "inspector_replay", "inputs": ["inspect_source.py", "source_index.json"], "worker_visible": False, "host_only": True},
    {"stage": "prepare", "code": "candidate_invalid", "inputs": ["cases.jsonl", "provenance.jsonl", "output_manifest.json"], "worker_visible": True, "host_only": False},
    {"stage": "prepare", "code": "submission_outcome", "inputs": ["submission.json"], "worker_visible": True, "host_only": False},
    {"stage": "prepare", "code": "accepted_artifact_immutable", "inputs": ["accepted handoff", "outputs"], "worker_visible": True, "host_only": False},
    {"stage": "prepare", "code": "population_reconciliation_failure", "inputs": ["image_population.jsonl", "provenance.jsonl"], "worker_visible": True, "host_only": False},
    {"stage": "prepare", "code": "execution_report_failure", "inputs": ["execution_report.json"], "worker_visible": True, "host_only": False},
    {"stage": "prepare", "code": "grouping_policy_failure", "inputs": ["grouping_policy.json", "mapping.json"], "worker_visible": True, "host_only": False},
    {"stage": "prepare", "code": "output_manifest_failure", "inputs": ["output_manifest.json"], "worker_visible": True, "host_only": False},
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_population_records(path) -> list[PopulationRecord]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(PopulationRecord.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"Invalid population record at line {line_number}: {exc}") from exc
    return records


def build_population_summary(records: Iterable[PopulationRecord], *, unit_type: str | None = None) -> PopulationSummary:
    """Build the report projection from one canonical population sequence."""
    rows = list(records)
    if not rows:
        raise ValueError("Cannot summarize an empty population")
    observed_types = {row.unit_type for row in rows}
    summary_type = unit_type or (next(iter(observed_types)) if len(observed_types) == 1 else "mixed")
    labels = Counter(row.label_status for row in rows)
    return PopulationSummary(
        unit_count=len(rows), unit_type=summary_type,
        labeled_count=labels.get("labeled", 0), unlabeled_count=labels.get("unlabeled", 0),
        source_counts=dict(sorted(Counter(row.source_id for row in rows).items())),
    )


def write_population_projection(
    records: Iterable[PopulationRecord], *, report: dict[str, Any], output_dir: str | Path,
) -> dict[str, Any]:
    """Write image_population.jsonl and source_report.json from one row sequence."""
    rows = list(records)
    summary = build_population_summary(rows)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "image_population.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8",
    )
    projected = dict(report)
    projected["population"] = summary.model_dump(mode="json")
    (root / "source_report.json").write_text(
        json.dumps(projected, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return projected


def _save(result, state, candidate):
    root = Path(state.get("preparation_root") or Path(candidate).parent)
    attempt_id = state.get("attempt_id") or result.stage.value
    receipt = root / "validation" / f"{attempt_id}.json"
    result = result.model_copy(update={"receipt_ref": str(receipt)})
    atomic_write_json(receipt, result.model_dump(mode="json"))
    return result


def read_question_set(path, *, stage):
    questions = QuestionSet.model_validate_json(Path(path).read_text(encoding="utf-8")).questions
    if any(item.stage is not stage for item in questions):
        raise ValueError("Question stage does not match the assigned stage")
    return [item.model_dump(mode="json") for item in questions]


def build_digest_ledger(root, cached=None):
    root = Path(root)
    cached = cached or {}
    ledger = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        stat = item.stat()
        previous = cached.get(relative)
        if isinstance(previous, dict):
            previous = DigestEntry.model_validate(previous, strict=False)
        if (isinstance(previous, DigestEntry)
                and previous.size_bytes == stat.st_size
                and previous.mtime_ns == stat.st_mtime_ns):
            ledger[relative] = previous
        else:
            ledger[relative] = DigestEntry(
                digest=digest_file(item), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            )
    return ledger


def candidate_digest(root, ledger=None):
    current = build_digest_ledger(root, ledger)
    return digest_payload({name: item.digest for name, item in current.items()})


def _semantic_control(issues):
    return (
        CONTROL_MAPPING_REVISION_REQUIRED
        if any(item.code in SEMANTIC_REVISION_CODES for item in issues)
        else None
    )


def _prepare_failure_code(message: str) -> str:
    if message.startswith("Prepare changed an accepted file:"):
        return "accepted_artifact_immutable"
    if (message.startswith("duplicate provenance population identity")
            or message.startswith("provenance references unknown population identity")
            or message.startswith("population coverage incomplete")):
        return "population_reconciliation_failure"
    if message.startswith("Execution report requires"):
        return "execution_report_failure"
    if message.startswith("Grouping policy differs"):
        return "grouping_policy_failure"
    if message.startswith("Prepare gate requires"):
        return "submission_outcome"
    return "candidate_invalid"


def validate_understand_candidate(
    outputs: dict[str, str], *, known_source_ids: Iterable[str] = (),
) -> ValidationResult:
    """Validate worker-visible Understand invariants without host side effects."""
    stage = Stage.UNDERSTAND
    issues: list[ValidationIssue] = []
    required = STAGE_SPECS[stage.value].required_outputs
    missing = [name for name in required if name not in outputs or not Path(outputs[name]).is_file()]
    if missing:
        issues.append(_issue(
            stage, "missing_output", f"Missing Understand outputs: {missing}",
            path="outputs", expected="all required outputs", observed=str(missing),
        ))
        return ValidationResult(status="rejected", stage=stage, issues=issues)

    try:
        report = SourceReport.model_validate_json(Path(outputs["source_report.json"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError) as exc:
        report = None
        issues.append(_issue(stage, "source_report_schema", str(exc), path="source_report.json"))
    if report is not None and (not report.sources or not report.checks or not report.population):
        issues.append(_issue(
            stage, "source_report_incomplete",
            "Source report requires measured sources, checks and population",
            path="source_report.json",
        ))

    try:
        mapping = PreparationMapping.model_validate_json(Path(outputs["mapping.json"]).read_text(encoding="utf-8"))
        if not mapping.fields or not mapping.population_policy or not mapping.case_identity or not mapping.grouping:
            issues.append(_issue(stage, "mapping_incomplete", "Mapping requires population policy, Case identity, grouping and field rules", path="mapping.json"))
    except (OSError, ValueError, ValidationError) as exc:
        issues.append(_issue(stage, "mapping_schema", str(exc), path="mapping.json"))

    try:
        ImageConversionPlan.model_validate_json(Path(outputs["image_conversion_plan.json"]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        issues.append(_issue(stage, "image_conversion_plan_invalid", str(exc)))

    records: list[PopulationRecord] = []
    try:
        records = read_population_records(outputs["image_population.jsonl"])
    except (OSError, ValueError) as exc:
        issues.append(_issue(stage, "population_schema", str(exc), path="image_population.jsonl"))
    if not records:
        issues.append(_issue(stage, "population_empty", "Population must contain at least one source unit", path="image_population.jsonl"))
    identities = [(record.source_id, record.source_unit_id) for record in records]
    if len(set(identities)) != len(identities):
        issues.append(_issue(stage, "population_duplicate_identity", "Population requires unique source-unit identities", path="image_population.jsonl"))
    known = set(known_source_ids)
    unknown = sorted({record.source_id for record in records if known and record.source_id not in known})
    if unknown:
        issues.append(_issue(stage, "population_unknown_source", f"Population references unknown sources: {unknown}", path="image_population.jsonl", expected=str(sorted(known)), observed=str(unknown)))
    if report is not None and records and report.population.unit_count != len(records):
        issues.append(_issue(
            stage, "population_count_mismatch",
            "Source report population count differs from population",
            path="source_report.json#/population/unit_count",
            expected=f"image_population rows={len(records)}",
            observed=f"report.population.unit_count={report.population.unit_count}",
            details={"reported_unit_count": report.population.unit_count, "population_rows": len(records)},
        ))

    try:
        questions = QuestionSet.model_validate_json(Path(outputs["semantic_questions.json"]).read_text(encoding="utf-8"))
        if any(item.stage is not stage for item in questions.questions):
            issues.append(_issue(stage, "question_stage_mismatch", "Question stage does not match Understand", path="semantic_questions.json"))
    except (OSError, ValueError, ValidationError) as exc:
        questions = None
        issues.append(_issue(stage, "question_schema", str(exc), path="semantic_questions.json"))

    try:
        submission = Submission.model_validate_json(Path(outputs["submission.json"]).read_text(encoding="utf-8"))
        if submission.requested_outcome != "submitted":
            issues.append(_issue(stage, "submission_outcome", "Understand gate requires a submitted candidate", path="submission.json", expected="submitted", observed=submission.requested_outcome))
    except (OSError, ValueError, ValidationError) as exc:
        submission = None
        issues.append(_issue(stage, "submission_schema", str(exc), path="submission.json"))

    if questions is not None:
        blocking = [item for item in questions.questions if item.blocking and not item.resolved]
        if blocking and not issues:
            return ValidationResult(status="needs_input", stage=stage, issues=issues, questions=blocking)
    return ValidationResult(status="rejected" if issues else "accepted", stage=stage, issues=issues)


def replay_inspector(outputs, *, state):
    """One host replay using host-visible sources, including Docker submissions."""
    script = Path(outputs["inspect_source.py"]).resolve()
    workspace = script.parent.parent
    with tempfile.TemporaryDirectory(dir=workspace / "work") as temporary:
        replay_root = Path(temporary)
        config = _json(workspace / "inputs/inspection_config.json")
        for source in config["sources"]:
            source["root"] = state["source_roots"][source["source_id"]]
            source["manifest"] = state["source_manifests"][source["source_id"]]
        config["source_manifests"] = state["source_manifests"]
        config_path = replay_root / "inspection_config.json"
        atomic_write_json(config_path, config)
        environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        completed = subprocess.run(
            [sys.executable, str(script), "--input-config", str(config_path),
             "--output-dir", str(replay_root), "--mode", "inventory"],
            cwd=workspace, env=environment, capture_output=True, text=True,
            encoding="utf-8", timeout=300, check=False,
        )
        if completed.returncode:
            raise ValueError(f"Inspector replay failed: {completed.stderr[-1000:]}")
        for name in ("source_report.json", "image_population.jsonl"):
            file = replay_root / name
            if not file.is_file() or file.read_bytes() != Path(outputs[name]).read_bytes():
                raise ValueError(f"Inspector replay does not match submitted {name}")


def validate_handoff(outputs, *, state=None, replay=True):
    state = state or {}
    root = Path(outputs) if isinstance(outputs, (str, Path)) else Path(outputs["submission.json"]).parent
    if isinstance(outputs, (str, Path)):
        outputs = {item.relative_to(root).as_posix(): str(item)
                   for item in root.rglob("*") if item.is_file()}
    stage = Stage.UNDERSTAND
    known_source_ids = (state.get("source_roots") or {}).keys()
    result = validate_understand_candidate(outputs, known_source_ids=known_source_ids)
    if result.status == "needs_input":
        return _save(result, state, root)
    if result.status != "accepted":
        return _save(result, state, root)
    try:
        if replay:
            replay_inspector(outputs, state=state)
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
        result = ValidationResult(
            status="rejected", stage=stage,
            issues=[_issue(stage, "inspector_replay", str(exc))],
        )
    return _save(result, state, root)


def create_accepted_handoff(state, outputs, validation):
    if validation.status != "accepted":
        raise ValueError("Understand validation must be accepted before creating a handoff")
    fields = {
        "source_report": "source_report.json", "mapping": "mapping.json",
        "population": "image_population.jsonl", "inspection_script": "inspect_source.py",
        "image_conversion_plan": "image_conversion_plan.json",
    }
    refs = {field: ArtifactRef(name=name, path=str(Path(outputs[name]).resolve()),
                              digest=digest_file(outputs[name]), producer="host",
                              stage=Stage.UNDERSTAND.value, status="accepted")
            for field, name in fields.items()}
    payload = {
        "dataset_id": state["dataset_id"], "source_snapshot_digest": state["source_snapshot_digest"],
        **refs,
        "decisions": [UserDecision.model_validate(item) for item in state.get("user_decisions", [])],
        "validation_receipt_ref": validation.receipt_ref,
    }
    digest = digest_payload({
        **payload, **{key: value.model_dump(mode="json") for key, value in refs.items()},
        "decisions": [item.model_dump(mode="json") for item in payload["decisions"]],
    })
    handoff = AcceptedHandoff(handoff_id=digest[:32], digest=digest, **payload)
    atomic_write_json(Path(state["preparation_root"]) / "handoff" / f"{digest}.json",
                      handoff.model_dump(mode="json"))
    return handoff


def validate_prepare_candidate(candidate, *, state=None, population_path=None, persist=True):
    """Read-only validation of a fully assembled candidate."""
    state = state or {}
    stage = Stage.PREPARE_IMAGE
    root = Path(candidate)
    try:
        report = validate_candidate(root, source_roots=state.get("source_roots"), population_path=population_path)
        issues = [_issue(stage, item["kind"], item["error"]) for item in report["failures"]]
        ledger = build_digest_ledger(root)
        result = ValidationResult(
            stage=stage, status="accepted" if report["accepted_cases"] and not report["systemic_failure"] else "rejected",
            report=report, issues=issues, candidate_digest=candidate_digest(root, ledger), digest_ledger=ledger,
        )
    except (ValueError, KeyError, OSError) as exc:
        result = ValidationResult(stage=stage, status="rejected", issues=[_issue(stage, "candidate_invalid", str(exc))])
    return _save(result, state, root) if persist else result


def project_grouping(mapping: dict[str, Any] | PreparationMapping, *, mapping_digest=""):
    payload = mapping.model_dump(mode="json") if isinstance(mapping, PreparationMapping) else mapping
    grouping = payload["grouping"]
    return {
        "rule_id": grouping["rule_id"], "case_unit": grouping["case_unit"],
        "key_fields": list(grouping["key_fields"]),
        "conflict_policy": grouping.get("conflict_policy", "error"),
        "mapping_digest": mapping_digest or digest_payload(payload),
    }


def build_execution_report(execution_records: Iterable[dict[str, Any]], coverage, accounting, *,
                           mapping_digest="", population_digest="", input_digest=""):
    records = list(execution_records)
    tests = [item for item in records if item.get("kind") in {"test", "sample"}]
    full = next((item for item in records if item.get("kind") == "full"), None)
    if not tests or full is None or any(item.get("status") != "succeeded" for item in [*tests, full]):
        raise ValueError("Successful sample/test and full execution records are required")
    return {
        "mapping_digest": mapping_digest, "population_digest": population_digest, "input_digest": input_digest,
        "tests": tests, "full_execution": full, "mapping_rule_coverage": coverage,
        "population_accounting": accounting, "failed_checks": [], "limitations": [],
    }
