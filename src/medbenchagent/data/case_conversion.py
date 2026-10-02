"""Deterministic Case/Evidence construction helpers."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
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
                if isinstance(value, dict) and value.get("type") in {"value", "text", "region"}:
                    kind, raw_value = value["type"], value.get("value")
                    refs, unit, time = value.get("ref", []), value.get("unit"), value.get("time")
                else:
                    kind, raw_value, refs, unit, time = ("text" if isinstance(value, str) else "value"), value, [], None, None
                evidence.append(Evidence(id=f"e{index + 1}-{name}", type=kind, name=name,
                                         value=raw_value, ref=refs, unit=unit, time=time))
        case = CaseRecord(case_id=case_id, patient_id=patient_id, evidence=evidence)
        output.append(validate_case(case, asset_map))
    return output


def _v4_value_evidence(evidence: list[Evidence], value: Any, *, name: str, ref: list[str]) -> None:
    """Flatten one v4 annotation field without interpreting its medical meaning."""
    if value is None:
        return
    if isinstance(value, dict) and {"type", "value"} <= value.keys():
        kind = value["type"]
        raw = value["value"]
        explicit_ref = value.get("ref", ref)
        unit = value.get("unit")
        time = value.get("time")
    else:
        text_field = any(token in name.casefold() for token in ("report", "description", "narrative", "text", "note"))
        kind = "text" if isinstance(value, str) and text_field else "value"
        raw, explicit_ref, unit, time = value, ref, None, None
    if kind not in {"value", "text", "region"}:
        raise ValueError(f"Unsupported v4 Evidence type for {name}: {kind}")
    evidence.append(Evidence(id=stable_id(name, len(evidence), prefix="evidence"), type=kind,
                             name=name, value=raw, ref=list(explicit_ref or []), unit=unit, time=time))


def convert_v4_record(record: dict[str, Any], asset: AssetEntry, *, source_id: str = "source",
                      converter_version: str = "v5") -> CaseRecord:
    """Convert one entity-scoped annotation.v4 record to one canonical v5 Case."""
    if not isinstance(record, dict) or not isinstance(record.get("image"), dict):
        raise ValueError("annotation.v4 record requires an image entity")
    image = record["image"]
    raw_case = record.get("case", {}).get("case_id") or record.get("study", {}).get("study_id")
    raw_case = raw_case or image.get("image_id") or image.get("image_path")
    if not raw_case:
        raise ValueError("annotation.v4 record has no stable Case identity")
    case_id = stable_id(converter_version, source_id, raw_case, prefix="case")
    asset_evidence = Evidence(id="asset-1", type="asset", name="image", value=asset.asset_id)
    evidence = [asset_evidence]

    patient = record.get("patient", {})
    patient_id = patient.get("patient_id")
    for entity_name in ("patient", "case", "study"):
        entity = record.get(entity_name, {})
        for name, value in (entity.get("annotation") or {}).items():
            _v4_value_evidence(evidence, value, name=f"{entity_name}.{name}", ref=[])

    image_annotation = image.get("annotation") or {}
    for name, value in image_annotation.items():
        if name == "findings" and isinstance(value, list):
            for index, finding in enumerate(value):
                if not isinstance(finding, dict):
                    continue
                prefix = f"finding.{finding.get('finding_id', index + 1)}"
                region_id = None
                if finding.get("bbox") is not None:
                    region = Evidence(id=stable_id(prefix, "region", prefix="evidence"), type="region",
                                      name=f"{prefix}.region", value=finding["bbox"], ref=[asset_evidence.id])
                    evidence.append(region)
                    region_id = region.id
                for key, raw in finding.items():
                    if key in {"finding_id", "bbox", "provenance"}:
                        continue
                    _v4_value_evidence(evidence, raw, name=f"{prefix}.{key}",
                                       ref=[region_id] if region_id else [asset_evidence.id])
        else:
            _v4_value_evidence(evidence, value, name=f"image.{name}", ref=[asset_evidence.id])
    case = CaseRecord(case_id=case_id, patient_id=str(patient_id) if patient_id is not None else None,
                      evidence=evidence)
    return validate_case(case, {asset.asset_id: asset})


def convert_v4_jsonl(cases_path: str | Path, assets: Iterable[AssetEntry], output_path: str | Path,
                     *, source_id: str = "source", converter_version: str = "v5") -> int:
    """Convert an annotation.v4 JSONL file deterministically and return Case count."""
    by_id = {asset.asset_id: asset for asset in assets}
    by_path = {asset.source_path: asset for asset in assets}
    count = 0
    with Path(cases_path).open(encoding="utf-8") as source, Path(output_path).open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            image = record.get("image", {})
            asset = by_id.get(image.get("image_id")) or by_path.get(image.get("image_path"))
            if asset is None:
                raise ValueError(f"No Asset for v4 image on line {line_number}")
            target.write(convert_v4_record(record, asset, source_id=source_id,
                                           converter_version=converter_version).model_dump_json() + "\n")
            count += 1
    if not count:
        raise ValueError("annotation.v4 JSONL is empty")
    return count
