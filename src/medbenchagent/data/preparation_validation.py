"""Host-side validation for an annotation preparation candidate."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.asset_manifest import iter_asset_manifest
from medbenchagent.data.models import CaseRecord, validate_case
from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest, load_manifest
from medbenchagent.data.sources import digest_file


def validate_candidate(candidate: str | Path, *, source_root: str | Path | None = None,
                          source_roots: dict[str, str | Path] | None = None) -> dict:
    """Validate every Case and return a complete acceptance/quarantine report."""
    candidate = Path(candidate)
    cases_path = candidate / "cases.jsonl"
    manifest = candidate / "asset_manifest.jsonl"
    if not cases_path.is_file() or not manifest.is_file():
        raise ValueError("cases.jsonl and asset_manifest.jsonl are required")
    provenance = candidate / "provenance.jsonl"
    if not provenance.is_file():
        raise ValueError("provenance.jsonl is required")
    required_receipts = ("conversion_config.json", "grouping_policy.json", "mapping.json", "complete.json")
    missing_receipts = [name for name in required_receipts if not (candidate / name).is_file()]
    if missing_receipts:
        raise ValueError("Candidate is missing required outputs: " + ", ".join(missing_receipts))
    provenance_failures = []
    provenance_count = 0
    provenance_cases: set[str] = set()
    with provenance.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not item.get("source_id") or not item.get("record_ref"):
                    raise ValueError("source_id and record_ref are required")
                if item.get("case_id"):
                    provenance_cases.add(str(item["case_id"]))
                roots = source_roots or ({"source": source_root} if source_root is not None else {})
                root = roots.get(str(item["source_id"]))
                for locator in item.get("locators", []):
                    if not isinstance(locator, dict) or not locator.get("path"):
                        raise ValueError("provenance locators require a path")
                    if root is not None:
                        path = Path(locator["path"])
                        resolved = (Path(root).resolve() / path).resolve()
                        if path.is_absolute() or not resolved.is_relative_to(Path(root).resolve()):
                            raise ValueError("provenance locator escapes source root")
                        if not resolved.is_file():
                            raise ValueError(f"provenance locator is missing: {locator['path']}")
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
    return {
        "schema_version": "preparation-report",
        "status": "failed" if failures and not cases else "complete",
        "input_units": len(cases) + len(failures),
        "input_cases": len(cases) + len(failures),
        "accepted_cases": len(cases),
        "quarantined_cases": len(failures),
        "failures": [*failures, *provenance_failures, *manifest_failures],
        "assets": len(assets),
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

