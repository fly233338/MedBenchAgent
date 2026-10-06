"""Two host gates: accept Understand semantics, then validate Prepare data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.data.preparation_validation import validate_candidate
from medbenchagent.runtime.run_artifacts import atomic_write_json

from .contracts import (
    AcceptedHandoff, ArtifactRef, PreparationMapping, QuestionSet, Stage, SourceReport,
    Submission, ValidationIssue, ValidationResult, STAGE_SPECS, UserDecision,
    digest_file, digest_payload,
)


def _issue(stage, code, message, **values):
    return ValidationIssue(stage=stage, code=code, message=message,
                           issue_id=f"{stage.value}:{code}:{digest_payload(message)[:16]}", **values)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def candidate_digest(root):
    root = Path(root)
    return digest_payload({item.relative_to(root).as_posix(): digest_file(item)
                           for item in sorted(root.rglob("*")) if item.is_file()})


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
    try:
        missing = [name for name in STAGE_SPECS[stage.value].required_outputs
                   if name not in outputs or not Path(outputs[name]).is_file()]
        if missing:
            raise ValueError(f"Missing Understand outputs: {missing}")
        report = SourceReport.model_validate_json(Path(outputs["source_report.json"]).read_text(encoding="utf-8"))
        if not report.sources or not report.checks or not report.population:
            raise ValueError("Source report requires measured sources, checks and population")
        mapping = PreparationMapping.model_validate_json(Path(outputs["mapping.json"]).read_text(encoding="utf-8"))
        if not mapping.fields or not mapping.population_policy or not mapping.case_identity or not mapping.grouping:
            raise ValueError("Mapping requires population policy, Case identity, grouping and field rules")
        records = [json.loads(line) for line in Path(outputs["image_population.jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        identities = {(item["source_id"], item["source_unit_id"]) for item in records}
        if not records or len(identities) != len(records) or any(not item.get("locator") for item in records):
            raise ValueError("Population requires unique source-unit identities and locators")
        if any(item[0] not in state["source_roots"] for item in identities):
            raise ValueError("Population references an unknown source")
        if report.population.get("unit_count") != len(records):
            raise ValueError("Source report population count differs from the population")
        questions = read_question_set(outputs["semantic_questions.json"], stage=stage)
        submission = Submission.model_validate_json(Path(outputs["submission.json"]).read_text(encoding="utf-8"))
        if submission.requested_outcome != "submitted":
            raise ValueError("Understand gate requires a submitted candidate")
        blocking = [item for item in questions if item["blocking"] and not item["resolved"]]
        if blocking:
            from .contracts import SemanticQuestion
            return _save(ValidationResult(
                status="needs_input", stage=stage,
                questions=[SemanticQuestion.model_validate(item, strict=False) for item in blocking],
            ), state, root)
        if replay:
            replay_inspector(outputs, state=state)
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as exc:
        return _save(ValidationResult(status="rejected", stage=stage,
                                     issues=[_issue(stage, "understand_invalid", str(exc))]), state, root)
    return _save(ValidationResult(status="accepted", stage=stage), state, root)


def create_accepted_handoff(state, outputs, validation):
    if validation.status != "accepted":
        raise ValueError("Understand validation must be accepted before creating a handoff")
    fields = {
        "source_report": "source_report.json", "mapping": "mapping.json",
        "population": "image_population.jsonl", "inspection_script": "inspect_source.py",
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


def validate_prepare_candidate(candidate, *, state=None, population_path=None):
    root, state, stage = Path(candidate).resolve(), state or {}, Stage.PREPARE
    try:
        missing = [name for name in STAGE_SPECS[stage.value].required_outputs if not (root / name).is_file()]
        if missing:
            raise ValueError(f"Missing Prepare outputs: {missing}")
        submission = Submission.model_validate_json((root / "submission.json").read_text(encoding="utf-8"))
        if "mapping_revision_required" in submission.diagnostic_codes:
            return _save(ValidationResult(
                status="blocked", stage=stage, control="mapping_revision_required",
                issues=[_issue(stage, "mapping_revision_required", submission.summary or "Mapping revision requested")],
            ), state, root)
        if submission.requested_outcome != "submitted":
            raise ValueError("Prepare gate requires a submitted candidate")
        if (root / "semantic_questions.json").is_file():
            from .contracts import SemanticQuestion
            questions = [item for item in read_question_set(root / "semantic_questions.json", stage=stage)
                         if item["blocking"] and not item["resolved"]]
            if questions:
                return _save(ValidationResult(status="needs_input", stage=stage,
                    questions=[SemanticQuestion.model_validate(item, strict=False) for item in questions]), state, root)
        handoff_value = state.get("accepted_handoff")
        handoff = AcceptedHandoff.model_validate(handoff_value, strict=False) if handoff_value else None
        if handoff:
            # Packaging restores the accepted semantic files. A worker-modified
            # mapping is an implementation error, never implicit acceptance.
            import shutil
            for artifact in (handoff.source_report, handoff.mapping, handoff.population, handoff.inspection_script):
                target = root / artifact.name
                if target.exists() and digest_file(target) != artifact.digest:
                    raise ValueError(f"Prepare changed an accepted file: {artifact.name}")
                if not target.exists():
                    shutil.copy2(artifact.path, target)
            population_file = Path(population_path or handoff.population.path)
            expected = set()
            for line in population_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    expected.add((str(item["source_id"]), str(item["source_unit_id"])))
            accounted = set()
            for line_number, line in enumerate((root / "provenance.jsonl").read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                identity = (str(item.get("source_id", "")), str(item.get("source_unit_id", "")))
                if identity in accounted:
                    raise ValueError(f"duplicate provenance population identity at line {line_number}: {identity}")
                if identity not in expected:
                    raise ValueError(f"provenance references unknown population identity: {identity}")
                if item.get("status") not in {"retained", "quarantined", "excluded", "duplicate"}:
                    raise ValueError(f"provenance line {line_number} has no canonical terminal status")
                accounted.add(identity)
            if accounted != expected:
                raise ValueError(f"population coverage incomplete: expected {len(expected)}, accounted {len(accounted)}")
        execution = _json(root / "execution_report.json")
        if (not execution.get("tests") or not execution.get("full_execution")
                or execution["full_execution"].get("status") != "succeeded"
                or any(item.get("status") != "succeeded" for item in execution["tests"])
                or execution.get("failed_checks")):
            raise ValueError("Execution report requires successful sample/test and full execution")
        if handoff:
            if _json(root / "grouping_policy.json") != project_grouping(
                _json(handoff.mapping.path), mapping_digest=handoff.mapping.digest,
            ):
                raise ValueError("Grouping policy differs from the accepted mapping")
        report = validate_candidate(
            root, source_roots=state.get("source_roots"), population_path=population_path,
            expected_mapping_digest=handoff.mapping.digest if handoff else None,
            expected_population_digest=handoff.population.digest if handoff else None,
        )
    except (ValueError, KeyError, OSError) as exc:
        return _save(ValidationResult(status="rejected", stage=stage,
                                     issues=[_issue(stage, "candidate_invalid", str(exc))]), state, root)
    issues = [_issue(stage, item["kind"], item["error"]) for item in report["failures"]]
    accounting = {"expected": int(report.get("population", {}).get("expected_units", 0) or 0),
                  "accounted": int(report.get("population", {}).get("accounted_units", 0) or 0),
                  "retained": 0, "quarantined": 0, "excluded": 0, "duplicate": 0,
                  "unprocessed": int(report.get("population", {}).get("unprocessed_units", 0) or 0)}
    provenance = root / "provenance.jsonl"
    if provenance.is_file():
        for line in provenance.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("status") in accounting:
                accounting[json.loads(line)["status"]] += 1
    result = ValidationResult(
        status="accepted" if report["accepted_cases"] and not report["systemic_failure"] else "rejected",
        stage=stage,
        control=("mapping_revision_required" if any(
            "mapping" in (item.code + " " + item.message).lower() for item in issues
        ) else None),
        issues=issues, report=report, accounting=accounting,
        quarantined_cases=[item for item in report["failures"] if item["kind"] == "hard_case_failure"],
        candidate_digest=candidate_digest(root),
    )
    return _save(result, state, root)


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
