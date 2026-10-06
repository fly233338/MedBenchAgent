"""Canonical host validation for understand handoffs and prepare candidates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from medbenchagent.data.preparation_validation import validate_candidate as validate_data_candidate

from .contracts import (
    AcceptedHandoff,
    ArtifactRef,
    PreparationMapping,
    SemanticQuestion,
    Stage,
    ValidationIssue,
    ValidationResult,
    digest_file,
    digest_payload,
)


def _issue(stage: Stage, code: str, message: str, *, path: str | None = None, issue_id: str | None = None, **details: Any) -> ValidationIssue:
    issue_id = issue_id or f"{stage.value}:{code}:{digest_payload({'message': message, 'path': path, 'details': details})[:16]}"
    return ValidationIssue(issue_id=issue_id, code=code, stage=stage, message=message, path=path, details=details)


def read_question_set(path: str | Path, *, stage: Stage) -> list[dict[str, Any]]:
    """Read the one canonical question container; reject legacy shapes."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"semantic_questions.json is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"questions"} or not isinstance(payload["questions"], list):
        raise ValueError("semantic_questions.json must be an object with exactly one questions array")
    questions = []
    for item in payload["questions"]:
        if not isinstance(item, dict):
            raise ValueError("semantic_questions.questions entries must be objects")
        SemanticQuestion.model_validate(item, strict=False)
        if item.get("stage") != stage.value:
            raise ValueError(f"Question stage does not match {stage.value}: {item.get('issue_id', '')}")
        questions.append(item)
    ids = [str(item["issue_id"]) for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("semantic_questions contains duplicate issue_id values")
    return questions


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{name} is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _artifact_refs(outputs: dict[str, str], *, stage: Stage, status: str = "candidate") -> dict[str, ArtifactRef]:
    refs: dict[str, ArtifactRef] = {}
    for name, raw_path in outputs.items():
        path = Path(raw_path)
        if not path.is_file():
            continue
        refs[name] = ArtifactRef(
            name=name, path=str(path.resolve()), digest=digest_file(path),
            producer="worker" if status == "candidate" else "host", stage=stage.value, status=status,
        )
    return refs


def _strict_mapping(payload: dict[str, Any]) -> PreparationMapping:
    """Validate the canonical mapping while keeping the model reader central."""
    mapping = PreparationMapping.model_validate(payload, strict=False)
    if not mapping.fields:
        raise ValueError("mapping.fields must contain at least one typed rule")
    if not mapping.population_policy or not mapping.case_identity or not mapping.grouping:
        raise ValueError("mapping population_policy, case_identity and grouping are required")
    for name, value in (("population_policy", mapping.population_policy), ("case_identity", mapping.case_identity), ("grouping", mapping.grouping)):
        if isinstance(value, dict) and (not value.get("rule_id") or not value.get("evidence_refs")):
            raise ValueError(f"mapping.{name} must include rule_id and evidence_refs")
    for rule in mapping.fields:
        if isinstance(rule, dict):
            required = {"rule_id", "source_path", "disposition", "source_type", "target_type", "missingness", "reason", "scope", "evidence_refs"}
            missing = required - set(rule)
            if missing:
                raise ValueError(f"mapping field rule is missing {sorted(missing)}")
            if not rule["evidence_refs"]:
                raise ValueError(f"mapping field {rule['rule_id']} has no evidence_refs")
    return mapping


def _population_records(path: str | Path, *, source_ids: set[str] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not item.get("source_id") or not item.get("source_unit_id") or not item.get("locator"):
                raise ValueError(f"image_population line {line_number} requires source_id, source_unit_id and locator")
            identity = (str(item["source_id"]), str(item["source_unit_id"]))
            if identity in identities:
                raise ValueError(f"Duplicate population identity: {identity}")
            identities.add(identity)
            if source_ids and identity[0] not in source_ids:
                raise ValueError(f"Population references unknown source: {identity[0]}")
            records.append(item)
    if not records:
        raise ValueError("image_population.jsonl must contain at least one source unit")
    return records


def validate_handoff(
    outputs: dict[str, str] | str | Path,
    *,
    state: dict[str, Any] | None = None,
    replay: bool = False,
) -> ValidationResult:
    """Validate an understand candidate and return structured host issues."""
    if isinstance(outputs, (str, Path)):
        root = Path(outputs)
        outputs = {path.name if path.parent == root else path.relative_to(root).as_posix(): str(path) for path in root.rglob("*") if path.is_file()}
    state = state or {}
    required = ("source_report.json", "image_population.jsonl", "mapping.json", "semantic_questions.json", "inspect_source.py", "submission.json")
    issues: list[ValidationIssue] = []
    for name in required:
        if not Path(outputs.get(name, "")).is_file():
            issues.append(_issue(Stage.UNDERSTAND, "missing_output", f"Missing understand output: {name}", path=name))
    if issues:
        return ValidationResult(status="rejected", stage=Stage.UNDERSTAND, issues=issues)
    try:
        report = _load_json(outputs["source_report.json"], name="source_report.json")
        if not report.get("sources") or not report.get("checks") or not isinstance(report.get("population"), dict):
            raise ValueError("source_report must contain sources, checks and population observations")
        for index, source in enumerate(report["sources"], 1):
            if not isinstance(source, dict) or not source.get("id") or not source.get("formats") or "file_count" not in source:
                raise ValueError(f"source_report.sources[{index}] must identify formats and file_count")
        for index, check in enumerate(report["checks"], 1):
            if not isinstance(check, dict) or not check.get("id") or not check.get("reproduction"):
                raise ValueError(f"source_report.checks[{index}] must include id and reproduction")
        _strict_mapping(_load_json(outputs["mapping.json"], name="mapping.json"))
        source_ids = {str(item.get("id")) for item in state.get("sources", []) if isinstance(item, dict) and item.get("id")}
        population = _population_records(outputs["image_population.jsonl"], source_ids=source_ids or None)
        if "unit_count" in report["population"] and int(report["population"]["unit_count"]) != len(population):
            raise ValueError("source_report population.unit_count does not match image_population")
        questions = read_question_set(outputs["semantic_questions.json"], stage=Stage.UNDERSTAND)
        compile(Path(outputs["inspect_source.py"]).read_text(encoding="utf-8"), str(outputs["inspect_source.py"]), "exec")
        submission = _load_json(outputs["submission.json"], name="submission.json")
        if submission.get("requested_outcome") != "submitted":
            raise ValueError("validate_handoff requires a submitted understand candidate")
    except Exception as exc:
        return ValidationResult(status="rejected", stage=Stage.UNDERSTAND, issues=[_issue(Stage.UNDERSTAND, "contract_invalid", str(exc))])
    blocking = [item for item in questions if item.get("blocking") and not item.get("resolved")]
    if blocking:
        return ValidationResult(status="needs_input", stage=Stage.UNDERSTAND, issues=[_issue(Stage.UNDERSTAND, "needs_input", item["question"], path="semantic_questions.json", issue_id=item["issue_id"]) for item in blocking], questions=[SemanticQuestion.model_validate(item, strict=False) for item in questions])
    if replay:
        replay_inspector(outputs, state=state)
    return ValidationResult(
        status="accepted", stage=Stage.UNDERSTAND, issues=[],
        valid_case_ids=[f"{item['source_id']}:{item['source_unit_id']}" for item in population],
        artifact_digests={name: digest_file(path) for name, path in outputs.items() if Path(path).is_file()},
    )


def replay_inspector(outputs: dict[str, str], *, state: dict[str, Any] | None = None) -> None:
    """Run the submitted inspector twice and compare it with its handoff."""
    script = Path(outputs["inspect_source.py"])
    attempt_root = script.parent.parent
    config = attempt_root / "inputs" / "inspection_config.json"
    if not config.is_file():
        raise ValueError("inspection_config.json is required for host replay")
    (attempt_root / "work").mkdir(parents=True, exist_ok=True)
    source_roots = (state or {}).get("source_roots") or {}
    with tempfile.TemporaryDirectory(dir=attempt_root / "work") as first, tempfile.TemporaryDirectory(dir=attempt_root / "work") as second:
        results = []
        for output_dir in (Path(first), Path(second)):
            environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            if source_roots:
                environment["MEDBENCH_SOURCE_ROOT"] = str(next(iter(source_roots.values())))
            completed = subprocess.run(
                [sys.executable, str(script), "--input-config", str(config), "--output-dir", str(output_dir), "--mode", "inventory"],
                cwd=attempt_root, env=environment, capture_output=True, text=True, encoding="utf-8", check=False,
            )
            if completed.returncode:
                raise ValueError(f"Inspector replay failed with exit code {completed.returncode}: {completed.stderr[-1000:]}")
            results.append(output_dir)
        for name in ("source_report.json", "image_population.jsonl"):
            first_bytes = (results[0] / name).read_bytes() if (results[0] / name).is_file() else None
            second_bytes = (results[1] / name).read_bytes() if (results[1] / name).is_file() else None
            submitted = Path(outputs[name]).read_bytes() if name in outputs and Path(outputs[name]).is_file() else None
            if first_bytes is None or first_bytes != second_bytes or submitted != first_bytes:
                raise ValueError(f"Inspector replay does not match submitted {name}")


def validate_prepare_candidate(
    candidate: str | Path,
    *,
    state: dict[str, Any] | None = None,
    population_path: str | Path | None = None,
) -> ValidationResult:
    """Run the data-layer validator and normalize its result for graph use."""
    root = Path(candidate).resolve()
    state = state or {}
    issues: list[ValidationIssue] = []
    if state.get("canonical_validation"):
        population_file = Path(population_path or root / "image_population.jsonl")
        provenance_file = root / "provenance.jsonl"
        try:
            frozen = {(str(item["source_id"]), str(item["source_unit_id"])) for item in _population_records(population_file, source_ids=set((state.get("source_roots") or {}).keys()) or None)}
            outcomes = set()
            with provenance_file.open(encoding="utf-8", errors="strict") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    identity = (str(item.get("source_id", "")), str(item.get("source_unit_id", "")))
                    if identity in outcomes:
                        raise ValueError(f"duplicate provenance population identity at line {line_number}: {identity}")
                    if identity not in frozen:
                        raise ValueError(f"provenance references unknown population identity: {identity}")
                    if item.get("status") not in {"retained", "quarantined", "excluded", "duplicate"}:
                        raise ValueError(f"provenance line {line_number} has no canonical terminal status")
                    outcomes.add(identity)
            if outcomes != frozen:
                raise ValueError(f"population coverage incomplete: expected {len(frozen)}, accounted {len(outcomes)}")
        except Exception as exc:
            return ValidationResult(status="rejected", stage=Stage.PREPARE, issues=[_issue(Stage.PREPARE, "population_contract", str(exc))])
    try:
        report = validate_data_candidate(
            root,
            source_roots=state.get("source_roots"),
            population_path=population_path,
            expected_mapping_digest=(state.get("accepted_handoff") or {}).get("mapping_digest"),
        )
    except Exception as exc:
        return ValidationResult(status="rejected", stage=Stage.PREPARE, issues=[_issue(Stage.PREPARE, "candidate_invalid", str(exc))])
    for item in report.get("failures", []):
        issues.append(_issue(Stage.PREPARE, str(item.get("kind", "candidate_failure")), str(item.get("error", item)), path=item.get("manifest")))
    accepted = report.get("accepted_cases", 0)
    status = "accepted" if not report.get("systemic_failure") and accepted else "rejected"
    population_report = report.get("population", {})
    counts = {"expected": int(population_report.get("expected_units", 0) or 0),
              "accounted": int(population_report.get("accounted_units", 0) or 0),
              "retained": 0, "quarantined": 0, "excluded": 0, "duplicate": 0,
              "unprocessed": int(population_report.get("unprocessed_units", 0) or 0)}
    provenance = root / "provenance.jsonl"
    if provenance.is_file():
        for line in provenance.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("status") in counts:
                counts[json.loads(line)["status"]] += 1
    return ValidationResult(
        status=status, stage=Stage.PREPARE, issues=issues,
        valid_case_ids=[], quarantined_cases=[item for item in report.get("failures", []) if item.get("kind") == "hard_case_failure"],
        accounting=counts,
    )


def project_grouping(mapping: dict[str, Any] | PreparationMapping) -> dict[str, Any]:
    """Derive grouping policy from mapping; there is no second model-owned policy."""
    payload = mapping.model_dump(mode="json") if isinstance(mapping, PreparationMapping) else mapping
    grouping = payload.get("grouping")
    if not grouping:
        raise ValueError("mapping.grouping is required")
    return {
        "rule_id": grouping.get("rule_id", ""),
        "case_unit": grouping.get("case_unit", ""),
        "key_fields": list(grouping.get("key_fields") or []),
        "conflict_policy": grouping.get("conflict_policy", "error"),
        "mapping_digest": digest_payload(payload),
    }


def build_execution_report(execution_records: Iterable[dict[str, Any]], coverage: dict[str, Any], accounting: dict[str, int], *, mapping_digest: str = "", population_digest: str = "", input_digest: str = "") -> dict[str, Any]:
    records = list(execution_records)
    tests = [item for item in records if item.get("kind") in {"test", "sample"}]
    full = next((item for item in records if item.get("kind") == "full"), None)
    if full is None:
        raise ValueError("A full execution record is required")
    if not all(item.get("status") == "succeeded" for item in [*tests, full]):
        raise ValueError("Execution report cannot claim success with failed records")
    return {
        "mapping_digest": mapping_digest, "population_digest": population_digest, "input_digest": input_digest,
        "tests": tests, "full_execution": full, "mapping_rule_coverage": coverage,
        "population_accounting": accounting, "failed_checks": [], "limitations": [],
    }


def accepted_handoff_from_result(dataset_id: str, outputs: dict[str, str], *, source_snapshot_digest: str, decisions_digest: str, contract_digest: str, validation_receipt_ref: str) -> AcceptedHandoff:
    refs = _artifact_refs(outputs, stage=Stage.UNDERSTAND, status="accepted")
    required = {"source_report.json", "image_population.jsonl", "mapping.json", "inspect_source.py"} - set(refs)
    if required:
        raise ValueError(f"AcceptedHandoff is missing {sorted(required)}")
    handoff_id = digest_payload({"dataset_id": dataset_id, "source": source_snapshot_digest, "refs": {key: value.digest for key, value in refs.items()}, "decisions": decisions_digest})[:32]
    return AcceptedHandoff(
        handoff_id=handoff_id, dataset_id=dataset_id, source_snapshot_digest=source_snapshot_digest,
        mapping_digest=refs["mapping.json"].digest, population_digest=refs["image_population.jsonl"].digest,
        source_report_digest=refs["source_report.json"].digest, decisions_digest=decisions_digest,
        inspection_script_digest=refs["inspect_source.py"].digest, inspection_dependency_digests={},
        contract_digest=contract_digest, validation_receipt_ref=validation_receipt_ref,
        artifact_refs=refs, revision=handoff_id,
    )
