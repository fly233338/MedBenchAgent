"""Structural task facts derived only from explicit Case relations."""

from collections import Counter, defaultdict
from typing import Any, Iterable


def case_facts(case, assets):
    rows = case["evidence"]
    by_id = {row["id"]: row for row in rows}
    media = [row for row in rows if row["type"] == "asset"]
    image_ids = {row["id"] for row in media if assets[row["value"]]["kind"] == "image"}
    explicit_groups = [set(row.get("ref", [])) for row in rows]
    related_images = any(len(refs & image_ids) >= 2 for refs in explicit_groups)
    paired_text = any(row["type"] == "text" and any(by_id[k]["type"] == "asset" for k in row.get("ref", [])) for row in rows)
    comparable = any(len(refs) >= 2 and len({by_id[k]["type"] for k in refs}) == 1 for refs in explicit_groups)
    counts = Counter(row["type"] for row in rows)
    counts["numeric"] = sum(row["type"] == "value" and type(row["value"]) in (int, float) for row in rows)
    # Integer values are structurally index-like; the model must prove index semantics.
    counts["index"] = sum(row["type"] == "value" and type(row["value"]) is int and row["value"] >= 0 for row in rows)
    return {"evidence": dict(counts), "evidence_names": sorted({r["name"] for r in rows}),
            "assets": {"image_assets": len(image_ids), "multi_image_cases": int(len(media) >= 2),
                       "related_image_cases": int(related_images),
                       "volume_assets": sum(assets[row["value"]]["kind"] == "volume" for row in media)},
            "relations": {"paired": int(paired_text), "paired_study": int(comparable),
                          "comparable": 2 if comparable else 0,
                          "related": 2 if any(row.get("ref") for row in rows) else 0,
                          "paired_text": int(paired_text)}}


def merge_facts(facts):
    result = {"evidence": Counter(), "assets": Counter(), "relations": Counter(), "evidence_names": set()}
    for value in facts:
        for key in ("evidence", "assets", "relations"):
            result[key].update(value[key])
        result["evidence_names"].update(value["evidence_names"])
    return {key: sorted(value) if isinstance(value, set) else dict(value) for key, value in result.items()}


def select_representative_case(cases: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select one stable, bounded Case for model-visible planning context."""
    values = [case for case in cases if isinstance(case.get("case_id"), str) and case["case_id"]]
    if not values:
        return None

    def score(case):
        evidence = case.get("evidence", [])
        return (-len({row.get("name") for row in evidence}), -len(evidence), str(case["case_id"]))

    return min(values, key=score)


def build_dataset_facts(
    cases: Iterable[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    *,
    profile: dict[str, Any] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build the single deterministic fact view shared by planning and execution."""
    case_list = list(cases)
    facts = [case_facts(case, assets) for case in case_list]
    merged = merge_facts(facts)
    evidence_catalog: dict[str, dict[str, Any]] = {}
    if profile:
        evidence_catalog = {row["name"]: row for row in profile.get("evidence", []) if row.get("name")}
    if not evidence_catalog:
        by_name: dict[str, dict[str, Any]] = defaultdict(lambda: {"cases": 0, "types": Counter(), "units": Counter()})
        for case in case_list:
            names = set()
            for row in case.get("evidence", []):
                name = row.get("name")
                if not name:
                    continue
                names.add(name)
                by_name[name]["types"][row.get("type")] += 1
                if row.get("unit"):
                    by_name[name]["units"][row["unit"]] += 1
            for name in names:
                by_name[name]["cases"] += 1
        evidence_catalog = {
            name: {"name": name, "cases": data["cases"],
                   "types": dict(data["types"]), "units": dict(data["units"])}
            for name, data in by_name.items()
        }
    patient_ids = {case.get("patient_id") for case in case_list if case.get("patient_id")}
    result = {
        "schema_version": "dataset-facts",
        "revision": revision,
        "case_count": len(case_list),
        "patient_count": len(patient_ids) if patient_ids else len(case_list),
        "evidence_catalog": evidence_catalog,
        "annotation_mapping": [
            {"role": f"evidence_{index}", "name": name, "evidence_type": evidence_type}
            for index, (name, row) in enumerate(sorted(evidence_catalog.items()))
            for evidence_type in sorted(row.get("types", {}))
            if evidence_type in {"value", "text", "asset", "region"}
        ],
        "evidence": merged["evidence"],
        "evidence_names": merged["evidence_names"],
        "assets": merged["assets"],
        "relations": merged["relations"],
        "representative_case": select_representative_case(case_list),
    }
    return result
