"""Structural task facts derived only from explicit Case relations."""
from collections import Counter


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
