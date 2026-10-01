"""Catalog and Card contracts for immutable annotation.v5 bundles."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.sources import digest_file


def read_v5_card(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "dataset-card.v3":
        raise ValueError("Dataset Card requires dataset-card.v3")
    return value


def verify_v5_bundle(bundle: str | Path) -> dict:
    bundle = Path(bundle).resolve()
    ready = json.loads((bundle / "ready.json").read_text(encoding="utf-8"))
    if ready.get("annotation_schema") != "annotation.v5" or ready.get("schema_version") != "dataset-bundle.v4":
        raise ValueError("Bundle is not annotation.v5")
    access = ready["access"]
    for key in ("annotation_path", "asset_manifest_path", "profile_path", "dataset_card_path"):
        if not Path(access[key]).is_file():
            raise ValueError(f"Missing v5 bundle artifact: {key}")
    checks = {
        "annotation_digest": access["annotation_path"],
        "asset_manifest_digest": access["asset_manifest_path"],
        "profile_digest": access["profile_path"],
    }
    for digest_key, path in checks.items():
        if digest_file(path) != access[digest_key]:
            raise ValueError(f"v5 bundle digest mismatch: {digest_key}")
    card = read_v5_card(access["dataset_card_path"])
    if card["annotation_digest"] != access["annotation_digest"] or card["asset_manifest_digest"] != access["asset_manifest_digest"] or card["profile_digest"] != access["profile_digest"]:
        raise ValueError("Dataset Card facts do not match v5 bundle")
    return {"ready": ready, "card": card}
