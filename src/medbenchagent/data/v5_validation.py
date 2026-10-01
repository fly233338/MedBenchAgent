"""Host-side validation for an annotation.v5 preparation candidate."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.asset_manifest import iter_asset_manifest
from medbenchagent.data.v5_models import CaseRecord, validate_case


def validate_v5_candidate(candidate: str | Path, *, source_root: str | Path | None = None) -> dict:
    """Validate every Case and return a complete acceptance/quarantine report."""
    candidate = Path(candidate)
    annotation = candidate / "annotation.jsonl"
    manifest = candidate / "asset_manifest.jsonl"
    if not annotation.is_file() or not manifest.is_file():
        raise ValueError("annotation.jsonl and asset_manifest.jsonl are required")
    provenance = candidate / "provenance.jsonl"
    if not provenance.is_file():
        raise ValueError("provenance.jsonl is required")
    required_receipts = ("conversion_config.json", "grouping_policy.json", "mapping.json", "conversion_version", "complete.json")
    missing_receipts = [name for name in required_receipts if not (candidate / name).is_file()]
    if missing_receipts:
        raise ValueError("v5 candidate is missing required outputs: " + ", ".join(missing_receipts))
    provenance_failures = []
    provenance_count = 0
    with provenance.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not item.get("source_id") or not item.get("record_ref"):
                    raise ValueError("source_id and record_ref are required")
                provenance_count += 1
            except Exception as exc:
                provenance_failures.append({"line": line_number, "kind": "provenance_failure", "error": str(exc)})
    assets = {item.asset_id: item for item in iter_asset_manifest(manifest, source_root=source_root)}
    cases = []
    failures = []
    seen_cases = set()
    with annotation.open(encoding="utf-8") as handle:
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
        raise ValueError("annotation.jsonl is empty")
    if provenance_count != len(cases):
        provenance_failures.append({"kind": "provenance_failure", "error": "provenance count does not match accepted cases"})
    return {
        "schema_version": "preparation-report.v5",
        "status": "failed" if failures and not cases else "complete",
        "input_units": len(cases) + len(failures),
        "input_cases": len(cases) + len(failures),
        "accepted_cases": len(cases),
        "quarantined_cases": len(failures),
        "failures": [*failures, *provenance_failures],
        "assets": len(assets),
        "systemic_failure": bool(provenance_failures) or provenance_count == 0,
    }


def write_v5_report(candidate: str | Path, report: dict) -> Path:
    candidate = Path(candidate)
    path = candidate / "preparation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quarantined = [item for item in report.get("failures", []) if item.get("kind") == "hard_case_failure"]
    (candidate / "quarantine.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in quarantined), encoding="utf-8")
    (candidate / "optional_evidence_dropped.jsonl").write_text("", encoding="utf-8")
    return path
