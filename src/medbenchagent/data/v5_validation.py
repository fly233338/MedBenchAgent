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
    return {
        "schema_version": "preparation-report.v5",
        "status": "failed" if failures and not cases else "complete",
        "input_cases": len(cases) + len(failures),
        "accepted_cases": len(cases),
        "quarantined_cases": len(failures),
        "failures": failures,
        "assets": len(assets),
        "systemic_failure": False,
    }


def write_v5_report(candidate: str | Path, report: dict) -> Path:
    path = Path(candidate) / "preparation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
