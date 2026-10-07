"""Deterministic structural facts used by Task Planning.

This module deliberately describes observable dataset structure. It does not
infer medical labels or treat annotation masks as additional model inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def _asset_metadata(item: dict[str, Any], assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    asset_id = item.get("value") if item.get("type") == "asset" else None
    asset = assets.get(asset_id, {}) if isinstance(asset_id, str) else {}
    metadata = asset.get("metadata", {}) if isinstance(asset, dict) else {}
    return metadata if isinstance(metadata, dict) else {}


def _asset_role(item: dict[str, Any], assets: dict[str, dict[str, Any]]) -> str:
    """Return an explicit input/reference role without using names as hints."""
    if item.get("mask") is not None:
        return "reference"
    metadata = _asset_metadata(item, assets)
    visibility = metadata.get("input_visibility")
    role = metadata.get("semantic_role", metadata.get("role"))
    if visibility == "reference_only" or role in {"annotation", "mask", "label", "calibration", "reference"}:
        return "reference"
    if visibility == "model_visible" or role in {"primary", "model_input", "image", "volume"}:
        return "primary"
    # An unannotated image/volume is a possible visual input, but it never
    # contributes derived annotation capabilities without explicit metadata.
    return "primary" if item.get("type") == "asset" else "unknown"


def _is_mask_asset(item: dict[str, Any], assets: dict[str, dict[str, Any]]) -> bool:
    return item.get("type") == "asset" and item.get("mask") is not None


def _semantic_evidence_name(name: Any) -> bool:
    return isinstance(name, str) and bool(name.strip())


def case_facts(case: dict[str, Any], assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = case.get("evidence", [])
    by_id = {row["id"]: row for row in rows}
    asset_rows = [row for row in rows if row.get("type") == "asset"]
    image_rows = [
        row for row in asset_rows
        if row.get("value") in assets and assets[row["value"]].get("kind") == "image"
    ]
    primary_images = [row for row in image_rows if _asset_role(row, assets) == "primary"]
    annotation_images = [row for row in image_rows if _asset_role(row, assets) == "reference"]
    def source_identity(row):
        image = assets.get(row.get("value"), {})
        return (image["source_id"], image["source_path"]) if image.get("source_id") and image.get("source_path") else (row.get("value"),)
    primary_sources = {source_identity(row) for row in primary_images}
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
    # Index support comes from explicit view/sampling manifests, never from a
    # field name or an integer stored in an Evidence value.
    counts["index"] = 0
    mask_rows = [row for row in rows if _is_mask_asset(row, assets)]
    reference_only_names = {
        row.get("name") for row in rows
        if row.get("name") and (
            row.get("mask") is not None
            or _asset_role(row, assets) == "reference"
        )
    }
    mask_features = {
        feature: sum(feature in (row.get("mask") or {}) for row in mask_rows)
        for feature in (
            "bbox", "pixel_count", "area_fraction", "centroid", "component_count",
            "equivalent_diameter_pixels", "physical_area",
        )
    }
    plane_features = {
        feature: sum(
            any(feature in plane for plane in (row.get("mask") or {}).get("planes", []))
            for row in mask_rows
        )
        for feature in ("bbox", "pixel_count", "area_fraction", "centroid", "equivalent_diameter_pixels")
    }
    view_ids = {row.get("view_id") for row in primary_images if row.get("view_id")}
    panel_ids = {
        _asset_metadata(row, assets).get("panel_id")
        for row in primary_images
        if _asset_metadata(row, assets).get("panel_id")
    }
    return {
        "evidence": dict(counts),
        "evidence_names": sorted({r.get("name") for r in rows if _semantic_evidence_name(r.get("name")) and r.get("name")}),
        "assets": {
            "image_assets": len(image_rows),
            "primary_image_assets": len(primary_sources),
            "annotation_image_assets": len(annotation_images),
            "multi_image_cases": int(len(primary_sources) >= 2),
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
            "mask_to_bbox": len(mask_rows),
            "mask_geometry": {
                "coverage": len(mask_rows),
                **mask_features,
                "plane_features": plane_features,
            },
            "view_panel": {
                "view_identity": len(view_ids),
                "panel_identity": len(panel_ids),
                "ordered": 0,
            },
        },
        "mask_feature_coverage": mask_features | {f"plane_{key}": value for key, value in plane_features.items()},
        "view_panel_capabilities": {
            "view_identity": len(view_ids),
            "panel_identity": len(panel_ids),
            "ordered": 0,
        },
        "reference_only_evidence_names": sorted(reference_only_names),
    }


def _merge_capabilities(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit capability summaries without treating names as semantics."""
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict):
            merged[key] = _merge_capabilities(
                merged.get(key, {}) if isinstance(merged.get(key), dict) else {}, value
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[key] = merged.get(key, 0) + value
        elif isinstance(value, bool):
            merged[key] = bool(merged.get(key, False) or value)
        elif value is not None:
            merged[key] = value
    return merged


def merge_facts(facts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "evidence": Counter(),
        "assets": Counter(),
        "relations": Counter(),
        "derived_capabilities": {},
        "mask_feature_coverage": Counter(),
        "view_panel_capabilities": Counter(),
        "reference_only_evidence_names": set(),
        "evidence_names": set(),
    }
    for value in facts:
        for key in ("evidence", "assets", "relations"):
            result[key].update(value[key])
        for key in ("mask_feature_coverage", "view_panel_capabilities"):
            result[key].update(value.get(key, {}))
        result["derived_capabilities"] = _merge_capabilities(
            result["derived_capabilities"], value.get("derived_capabilities", {})
        )
        result["evidence_names"].update(value["evidence_names"])
        result["reference_only_evidence_names"].update(value.get("reference_only_evidence_names", []))
    return {
        key: sorted(value) if isinstance(value, set)
        else dict(value) if isinstance(value, Counter)
        else value
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
        "reference_only_evidence_names": merged["reference_only_evidence_names"],
        "assets": merged["assets"],
        "relations": merged["relations"],
        "derived_capabilities": merged["derived_capabilities"],
        "mask_feature_coverage": merged["mask_feature_coverage"],
        "view_panel_capabilities": merged["view_panel_capabilities"],
        "representative_case": select_representative_case(case_list),
    }



def compact_scalar(value: Any, *, limit: int = 180) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, list):
        return [compact_scalar(item, limit=limit) for item in value[:8]]
    if isinstance(value, dict):
        return {key: compact_scalar(item, limit=limit) for key, item in list(value.items())[:12]}
    return value


def _asset_rows(case: dict[str, Any], assets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in case.get("evidence", []):
        if item.get("type") != "asset" or item.get("value") not in assets:
            continue
        asset = assets[item["value"]]
        if asset.get("kind") not in {"image", "volume"}:
            continue
        if _asset_role(item, assets) != "primary":
            continue
        rows.append(item)
    return rows


def compact_case(
    case: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    *,
    names: set[str] | None = None,
    reference_names: set[str] | None = None,
) -> dict[str, Any]:
    primary = _asset_rows(case, assets)
    primary_ids = {item.get("id") for item in primary}
    rows = []
    for item in case.get("evidence", []):
        name = item.get("name")
        if item.get("type") == "asset" and item in primary:
            rows.append(item)
            continue
        if item.get("mask") is not None or (reference_names and name in reference_names):
            continue
        if reference_names is not None and item.get("type") in {"value", "text", "region"} and item.get("input_visibility") != "model_visible":
            continue
        if names and name not in names:
            continue
        refs = set(item.get("ref", []))
        if item.get("type") in {"value", "text", "region"} and (refs & primary_ids or not primary_ids):
            rows.append({**item, "value": compact_scalar(item.get("value"))})
    return {"evidence": rows[:20]}


def compact_catalog_entry(entry: dict[str, Any], meaning: str | None = None) -> dict[str, Any]:
    item = {key: entry[key] for key in ("name", "cases", "types", "units", "value_count") if key in entry}
    if entry.get("observed_values"):
        item["observed_values"] = [
            {**value, "value": compact_scalar(value.get("value"))}
            for value in entry["observed_values"][:8]
            if isinstance(value, dict)
        ]
    if meaning:
        item["meaning"] = meaning
    return item
