"""Deterministic Case/Evidence construction helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from medbenchagent.data.asset_manifest import stable_id
from medbenchagent.data.v5_models import AssetEntry, CaseRecord, Evidence, validate_case


def choose_case_key(row: dict[str, Any]) -> str:
    for key in ("study_id", "exam_id", "accession_id", "case_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return stable_id(row.get("source_id"), row.get("source_path"), prefix="case")


def build_case_records(rows: Iterable[dict[str, Any]], assets: Iterable[AssetEntry], *, converter_version="v5"):
    assets_by_path = {entry.source_path: entry for entry in assets}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[choose_case_key(row)].append(row)
    output = []
    asset_map = {entry.asset_id: entry for entry in assets_by_path.values()}
    for raw_key, members in sorted(grouped.items()):
        case_id = stable_id(converter_version, raw_key, prefix="case")
        evidence: list[Evidence] = []
        patient_ids = {str(row["patient_id"]) for row in members if row.get("patient_id") not in (None, "")}
        patient_id = next(iter(patient_ids)) if len(patient_ids) == 1 else None
        for index, row in enumerate(members):
            asset = assets_by_path.get(row.get("source_path"))
            if asset is None:
                raise ValueError(f"Missing Asset for source path: {row.get('source_path')}")
            evidence.append(Evidence(id=f"asset-{index + 1}", type="asset", name=row.get("view", "asset"), value=asset.asset_id))
            for name, value in row.get("evidence", {}).items():
                kind = "text" if isinstance(value, str) else "value"
                evidence.append(Evidence(id=f"e{index + 1}-{name}", type=kind, name=name, value=value))
        case = CaseRecord(case_id=case_id, patient_id=patient_id, evidence=evidence)
        output.append(validate_case(case, asset_map))
    return output
