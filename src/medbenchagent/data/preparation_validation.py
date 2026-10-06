"""Host-side validation for an annotation preparation candidate."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.asset_manifest import iter_asset_manifest
from medbenchagent.data.models import CaseRecord, validate_case
from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest, load_manifest
from medbenchagent.data.sources import digest_file


def validate_candidate(candidate: str | Path, *, source_root: str | Path | None = None,
                          source_roots: dict[str, str | Path] | None = None,
                          population_path: str | Path | None = None,
                          expected_mapping_digest: str | None = None,
                          expected_population_digest: str | None = None) -> dict:
    """Validate every Case and return a complete acceptance/quarantine report."""
    candidate = Path(candidate)
    cases_path = candidate / "cases.jsonl"
    manifest = candidate / "asset_manifest.jsonl"
    if not cases_path.is_file() or not manifest.is_file():
        raise ValueError("cases.jsonl and asset_manifest.jsonl are required")
    provenance = candidate / "provenance.jsonl"
    if not provenance.is_file():
        raise ValueError("provenance.jsonl is required")
    # complete.json is a host receipt written after this validation succeeds;
    # workers must not be able to claim publication by creating it themselves.
    required_receipts = ["conversion_config.json", "grouping_policy.json", "mapping.json"]
    if (candidate / "submission.json").is_file():
        required_receipts.extend(("execution_report.json", "output_manifest.json"))
    missing_receipts = [name for name in required_receipts if not (candidate / name).is_file()]
    if missing_receipts:
        raise ValueError("Candidate is missing required outputs: " + ", ".join(missing_receipts))
    provenance_failures = []
    provenance_count = 0
    provenance_cases: set[str] = set()
    accounted: set[tuple[str, str]] = set()
    accounting = {"retained": 0, "quarantined": 0, "excluded": 0, "duplicate": 0}
    with provenance.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
                if (not isinstance(item, dict) or not item.get("source_id")
                        or not item.get("source_unit_id") or not item.get("record_ref")):
                    raise ValueError("source_id, source_unit_id and record_ref are required")
                identity = (str(item["source_id"]), str(item["source_unit_id"]))
                if item.get("status") in accounting:
                    if identity in accounted:
                        raise ValueError(f"duplicate provenance population identity: {identity}")
                    accounted.add(identity)
                    accounting[item["status"]] += 1
                    if item["status"] == "retained" and not (item.get("case_ids") or item.get("case_id")):
                        raise ValueError("Retained source units require case_ids")
                if item.get("case_id"):
                    provenance_cases.add(str(item["case_id"]))
                roots = source_roots or ({"source": source_root} if source_root is not None else {})
                root = roots.get(str(item["source_id"]))
                for locator in item.get("locators", []):
                    if isinstance(locator, dict):
                        locator = locator.get("path")
                    if not isinstance(locator, str) or not locator:
                        raise ValueError("provenance locators require a source-relative path")
                    if root is not None:
                        path = Path(locator)
                        resolved = (Path(root).resolve() / path).resolve()
                        if path.is_absolute() or not resolved.is_relative_to(Path(root).resolve()):
                            raise ValueError("provenance locator escapes source root")
                        if not resolved.is_file():
                            raise ValueError(f"provenance locator is missing: {locator}")
                provenance_count += 1
            except Exception as exc:
                provenance_failures.append({"line": line_number, "kind": "provenance_failure", "error": str(exc)})
    assets = {item.asset_id: item for item in iter_asset_manifest(
        manifest, source_root=source_root, source_roots=source_roots
    )}
    cases = []
    failures = []
    seen_cases = set()
    with cases_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                case = CaseRecord.model_validate_json(line)
                if case.case_id in seen_cases:
                    raise ValueError(f"duplicate case_id: {case.case_id}")
                seen_cases.add(case.case_id)
                cases.append(validate_case(case, assets))
            except Exception as exc:
                failures.append({"line": line_number, "kind": "hard_case_failure", "error": str(exc)})
    if not cases and not failures:
        raise ValueError("cases.jsonl is empty")
    if provenance_cases:
        missing_cases = sorted({case.case_id for case in cases} - provenance_cases)
        if missing_cases:
            provenance_failures.append({"kind": "provenance_failure",
                                        "error": f"accepted cases missing provenance: {missing_cases[:10]}"})
    manifest_failures = []
    case_by_id = {case.case_id: case for case in cases}
    for name, model in (
        ("presentation_manifest.jsonl", PresentationRecord),
        ("split_manifest.jsonl", SplitManifest),
        ("sampling_manifest.jsonl", SamplingManifest),
    ):
        path = candidate / name
        if not path.is_file():
            continue
        try:
            parsed = load_manifest(path, model)
            if name == "presentation_manifest.jsonl":
                records = parsed if isinstance(parsed, list) else [parsed]
                seen = set()
                for record in records:
                    if record.case_id not in case_by_id:
                        raise ValueError(f"presentation references unknown case: {record.case_id}")
                    asset = assets.get(record.asset_id)
                    if asset is None:
                        raise ValueError(f"presentation references unknown asset: {record.asset_id}")
                    if record.input_digest != asset.file_digest:
                        raise ValueError(f"presentation input digest mismatch: {record.presentation_id}")
                    output = Path(record.path)
                    resolved_output = output.resolve() if output.is_absolute() else (candidate / output).resolve()
                    if not resolved_output.is_relative_to(candidate.resolve()):
                        raise ValueError(f"presentation path escapes candidate: {record.path}")
                    output = resolved_output
                    if not output.is_file() or digest_file(output) != record.output_digest:
                        raise ValueError(f"presentation output digest mismatch: {record.presentation_id}")
                    key = (record.case_id, record.panel_id, record.order)
                    if key in seen:
                        raise ValueError(f"duplicate presentation order: {key}")
                    seen.add(key)
            elif name == "sampling_manifest.jsonl":
                for unit in parsed.units:
                    case = case_by_id.get(unit.case_id)
                    if case is None:
                        raise ValueError(f"sampling references unknown case: {unit.case_id}")
                    evidence_ids = {item.id for item in case.evidence}
                    if not set(unit.ordered_evidence_ids) <= evidence_ids:
                        raise ValueError(f"sampling references unknown Evidence: {unit.sampling_unit_id}")
                    view_ids = {item.view_id for item in case.evidence if item.view_id}
                    if not set(unit.view_ids) <= view_ids:
                        raise ValueError(f"sampling references unknown view: {unit.sampling_unit_id}")
            else:
                for record in parsed.records:
                    if record.case_id not in case_by_id:
                        raise ValueError(f"split references unknown case: {record.case_id}")
        except Exception as exc:
            manifest_failures.append({"kind": "manifest_failure", "manifest": name, "error": str(exc)})
    # Validate registered outputs and their candidate-relative dependencies once.
    output_manifest = candidate / "output_manifest.json"
    if output_manifest.is_file():
        try:
            payload = json.loads(output_manifest.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
                raise ValueError("output_manifest.files must be a list")
            seen_output_paths = set()
            for entry in payload["files"]:
                item = entry
                relative = item.get("path")
                if not isinstance(relative, str) or not relative:
                    raise ValueError("output_manifest file entries require a path")
                if relative in seen_output_paths:
                    raise ValueError(f"output_manifest contains a duplicate path: {relative}")
                seen_output_paths.add(relative)
                if relative in {"output_manifest.json", "submission.json"}:
                    raise ValueError("output_manifest must not register itself or submission.json")
                path = Path(relative)
                resolved = path.resolve() if path.is_absolute() else (candidate / path).resolve()
                if not resolved.is_relative_to(candidate.resolve()) or not resolved.is_file():
                    raise ValueError(f"output_manifest references a missing or external file: {relative}")
                expected = item.get("digest")
                if not expected or digest_file(resolved) != expected:
                    raise ValueError(f"output_manifest digest mismatch: {relative}")
            required_files = ("cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl",
                              "conversion_config.json", "grouping_policy.json", "execution_report.json",
                              "scripts/prepare.py")
            missing_registered = [name for name in required_files if (candidate / name).is_file() and name not in seen_output_paths]
            if missing_registered:
                raise ValueError(f"output_manifest does not register required outputs: {missing_registered}")
            dependencies = payload.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError("output_manifest.dependencies must be a list")
            for dependency in dependencies:
                if not isinstance(dependency, dict) or not dependency.get("path"):
                    raise ValueError("output_manifest dependency entries require a path")
                dependency_path = Path(str(dependency["path"]))
                resolved_dependency = dependency_path.resolve() if dependency_path.is_absolute() else (candidate / dependency_path).resolve()
                if not resolved_dependency.is_relative_to(candidate.resolve()):
                    raise ValueError(f"output_manifest dependency escapes candidate: {dependency['path']}")
                if not resolved_dependency.exists():
                    raise ValueError(f"output_manifest dependency is missing: {dependency['path']}")
                expected_dependency = dependency.get("digest")
                if expected_dependency and resolved_dependency.is_file() and digest_file(resolved_dependency) != expected_dependency:
                    raise ValueError(f"output_manifest dependency digest mismatch: {dependency['path']}")
            if expected_mapping_digest and payload.get("mapping_digest") != expected_mapping_digest:
                raise ValueError("output_manifest mapping digest does not match accepted mapping")
            if expected_population_digest and payload.get("population_digest") != expected_population_digest:
                raise ValueError("output_manifest population digest does not match accepted population")
        except Exception as exc:
            manifest_failures.append({"kind": "output_manifest_failure", "manifest": "output_manifest.json", "error": str(exc)})

    # Reconcile the independent population by stable source-unit identity when
    # the understand stage supplied it.  Case count is intentionally not used
    # as a proxy: several source units may legitimately form one Case.
    population = Path(population_path) if population_path else candidate / "image_population.jsonl"
    population_ids: set[tuple[str, str]] = set()
    accounted: set[tuple[str, str]] = set()
    missing_population: list[tuple[str, str]] = []
    if population.is_file():
        try:
            with population.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict) or not item.get("source_id") or not item.get("source_unit_id"):
                        raise ValueError(f"population record {line_number} requires source_id and source_unit_id")
                    key = (str(item["source_id"]), str(item["source_unit_id"]))
                    if key in population_ids:
                        raise ValueError(f"duplicate population identity: {key}")
                    population_ids.add(key)
            with provenance.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict) or not item.get("source_id") or not item.get("source_unit_id"):
                        raise ValueError("provenance requires source_id and source_unit_id")
                    status = item.get("status")
                    if status is not None and status not in {"retained", "quarantined", "excluded", "duplicate"}:
                        raise ValueError(f"invalid population outcome status: {status}")
                    accounted.add((str(item["source_id"]), str(item["source_unit_id"])))
            missing_population = sorted(population_ids - accounted)
            if missing_population:
                provenance_failures.append({"kind": "population_reconciliation_failure", "error": f"Source units missing provenance: {missing_population[:20]}", "missing_count": len(missing_population)})
        except Exception as exc:
            provenance_failures.append({"kind": "population_reconciliation_failure", "error": str(exc)})
    if output_manifest.is_file() and cases:
        # New submissions must explicitly bind every accepted Case to at least
        # one provenance row, even if an older bundle did not carry that field.
        if provenance_cases:
            missing_cases = sorted({case.case_id for case in cases} - provenance_cases)
            if missing_cases:
                provenance_failures.append({"kind": "provenance_failure", "error": f"accepted cases missing provenance: {missing_cases[:20]}"})
    return {
        "schema_version": "preparation-report",
        "status": "failed" if (failures and not cases) or provenance_failures or manifest_failures else "complete",
        "input_units": len(cases) + len(failures),
        "input_cases": len(cases) + len(failures),
        "accepted_cases": len(cases),
        "quarantined_cases": len(failures),
        "failures": [*failures, *provenance_failures, *manifest_failures],
        "assets": len(assets),
        "population": {
                "expected_units": len(population_ids),
                "accounted_units": len(accounted),
                "unprocessed_units": len(missing_population),
            "provenance_rows": provenance_count,
        },
        "systemic_failure": bool(provenance_failures or manifest_failures) or provenance_count == 0,
    }


def write_validation_report(candidate: str | Path, report: dict) -> Path:
    candidate = Path(candidate)
    path = candidate / "preparation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quarantined = [item for item in report.get("failures", []) if item.get("kind") == "hard_case_failure"]
    (candidate / "quarantine.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in quarantined), encoding="utf-8")
    (candidate / "optional_evidence_dropped.jsonl").write_text("", encoding="utf-8")
    return path

