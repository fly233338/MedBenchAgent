"""Deterministic structural facts used by Task Planning.

This module deliberately describes observable dataset structure. It does not
infer medical labels or treat annotation masks as additional model inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


_ANNOTATION_NAME_HINTS = ("mask", "segmentation", "annotation", "label", "roi")


def _is_annotation_asset(item: dict[str, Any]) -> bool:
    name = str(item.get("name", "")).casefold()
    return name.startswith(_ANNOTATION_NAME_HINTS) or any(
        token in name for token in ("_mask", "_seg", "mask_", "seg_")
    )


def _asset_role(item: dict[str, Any]) -> str:
    return "annotation" if _is_annotation_asset(item) else "primary"


def _is_mask_asset(item: dict[str, Any]) -> bool:
    name = str(item.get("name", "")).casefold()
    return item.get("type") == "asset" and any(
        token in name for token in ("mask", "segmentation", "roi")
    )


def _semantic_evidence_name(name: Any) -> bool:
    """Exclude case-specific locator fields from model-visible planning facts."""
    value = str(name or "")
    return not value.casefold().startswith(("mask_filename:", "mask_region:"))


def case_facts(case: dict[str, Any], assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = case.get("evidence", [])
    by_id = {row["id"]: row for row in rows}
    asset_rows = [row for row in rows if row.get("type") == "asset"]
    image_rows = [
        row for row in asset_rows
        if row.get("value") in assets and assets[row["value"]].get("kind") == "image"
    ]
    primary_images = [row for row in image_rows if _asset_role(row) == "primary"]
    annotation_images = [row for row in image_rows if _asset_role(row) == "annotation"]
    primary_ids = {row["id"] for row in primary_images}

    explicit_groups = [set(row.get("ref", [])) for row in rows]
    related_image_groups = [
        refs & primary_ids for refs in explicit_groups
        if len(refs & primary_ids) >= 2
    ]
    paired_text = any(
        row.get("type") == "text"
        and any(by_id.get(ref, {}).get("type") == "asset" for ref in row.get("ref", []))
        for row in rows
    )
    comparable = any(
        len(refs) >= 2
        and len({by_id[ref].get("type") for ref in refs if ref in by_id}) == 1
        for refs in explicit_groups
    )
    region_asset_refs = sum(
        row.get("type") == "region" and bool(row.get("ref"))
        for row in rows
    )
    counts = Counter(row.get("type") for row in rows)
    counts["numeric"] = sum(
        row.get("type") == "value" and type(row.get("value")) in (int, float)
        for row in rows
    )
    counts["index"] = sum(
        row.get("type") == "value"
        and any(token in str(row.get("name", "")).casefold() for token in ("index", "slice", "frame"))
        and type(row.get("value")) is int
        and row.get("value", -1) >= 0
        for row in rows
    )
    return {
        "evidence": dict(counts),
        "evidence_names": sorted({r.get("name") for r in rows if _semantic_evidence_name(r.get("name")) and r.get("name")}),
        "assets": {
            "image_assets": len(image_rows),
            "primary_image_assets": len(primary_images),
            "annotation_image_assets": len(annotation_images),
            "multi_image_cases": int(len(primary_images) >= 2),
            "related_image_cases": int(bool(related_image_groups)),
            "volume_assets": sum(
                assets[row["value"]].get("kind") == "volume"
                for row in asset_rows if row.get("value") in assets
            ),
        },
        "relations": {
            "region_asset_refs": region_asset_refs,
            "paired_text": int(paired_text),
            "paired": int(paired_text),
            "comparable_groups": int(comparable),
            "comparable": 2 if comparable else 0,
            "related_image_groups": len(related_image_groups),
            "paired_study": 2 if comparable else 0,
            "temporal_links": 0,
        },
        "derived_capabilities": {
            "mask_to_bbox": int(any(_is_mask_asset(row) for row in rows)),
        },
    }


def merge_facts(facts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "evidence": Counter(),
        "assets": Counter(),
        "relations": Counter(),
        "derived_capabilities": Counter(),
        "evidence_names": set(),
    }
    for value in facts:
        for key in ("evidence", "assets", "relations"):
            result[key].update(value[key])
        result["evidence_names"].update(value["evidence_names"])
    return {
        key: sorted(value) if isinstance(value, set) else dict(value)
        for key, value in result.items()
    }


def select_representative_case(cases: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select one stable, bounded Case for model-visible planning context."""
    values = [
        case for case in cases
        if isinstance(case.get("case_id"), str) and case["case_id"]
    ]
    if not values:
        return None

    def score(case):
        evidence = case.get("evidence", [])
        return (-len({row.get("name") for row in evidence}), -len(evidence), str(case["case_id"]))

    selected = min(values, key=score)
    return {
        **selected,
        "evidence": [
            row for row in selected.get("evidence", [])
            if _semantic_evidence_name(row.get("name"))
        ],
    }


def build_dataset_facts(
    cases: Iterable[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, semantically explicit dataset fact view."""
    case_list = list(cases)
    facts = [case_facts(case, assets) for case in case_list]
    merged = merge_facts(facts)
    evidence_catalog: dict[str, dict[str, Any]] = {}
    if profile:
        evidence_catalog = {
            row["name"]: row
            for row in profile.get("evidence", [])
            if row.get("name") and _semantic_evidence_name(row.get("name"))
        }
    if not evidence_catalog:
        by_name: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cases": 0, "types": Counter(), "units": Counter()}
        )
        for case in case_list:
            names = set()
            for row in case.get("evidence", []):
                name = row.get("name")
                if not name or not _semantic_evidence_name(name):
                    continue
                names.add(name)
                by_name[name]["types"][row.get("type")] += 1
                if row.get("unit"):
                    by_name[name]["units"][row["unit"]] += 1
            for name in names:
                by_name[name]["cases"] += 1
        evidence_catalog = {
            name: {
                "name": name,
                "cases": data["cases"],
                "types": dict(data["types"]),
                "units": dict(data["units"]),
            }
            for name, data in by_name.items()
        }

    # Preserve the registered profile while enriching value-like evidence with
    # the deterministic vocabulary observed in the canonical cases.
    observed: dict[str, Counter] = defaultdict(Counter)
    for case in case_list:
        for row in case.get("evidence", []):
            if row.get("type") == "value" and row.get("name") and row.get("value") is not None:
                value = row.get("value")
                if isinstance(value, (str, int, float, bool)):
                    observed[row["name"]][str(value)] += 1
    for name, counts in observed.items():
        entry = evidence_catalog.setdefault(name, {"name": name, "cases": 0, "types": {"value": 0}, "units": {}})
        entry["observed_values"] = [
            {"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        entry["value_count"] = len(counts)

    known_patient_ids = {
        case.get("patient_id") for case in case_list if case.get("patient_id")
    }
    return {
        "schema_version": "dataset-facts",
        "revision": revision,
        "case_count": len(case_list),
        "patient_count": len(known_patient_ids),
        "patient_identity": "present" if known_patient_ids else "absent",
        "evidence_catalog": evidence_catalog,
        "annotation_mapping": [
            {
                "role": f"{name}:{evidence_type}",
                "name": name,
                "evidence_type": evidence_type,
            }
            for name, row in sorted(evidence_catalog.items())
            for evidence_type in sorted(row.get("types", {}))
            if evidence_type in {"value", "text", "asset", "region"}
        ],
        "evidence": merged["evidence"],
        "evidence_names": merged["evidence_names"],
        "assets": merged["assets"],
        "relations": merged["relations"],
        "derived_capabilities": merged["derived_capabilities"],
        "representative_case": select_representative_case(case_list),
    }

