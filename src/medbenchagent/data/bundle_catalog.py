"""Catalog and Card contracts for immutable annotation bundles."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.sources import digest_file
from medbenchagent.data.manifests import PresentationRecord, SamplingManifest, SplitManifest, load_manifest


def read_card(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "dataset-card":
        raise ValueError("Dataset Card requires dataset-card")
    return value


def card_summary(card: dict) -> dict:
    """Return the bounded, model-visible summary for a validated Card."""
    if not isinstance(card, dict) or card.get("schema_version") != "dataset-card":
        raise ValueError("Dataset Card requires dataset-card")
    summary = card.get("summary", {})
    return {
        "dataset_id": card.get("dataset_id"),
        "name": card.get("name"),
        "description": card.get("description", ""),
        "case_count": card.get("case_count", summary.get("case_count", 0)),
        "patient_count": card.get("patient_count", summary.get("patient_count", 0)),
        "summary": summary,
        "semantics": card.get("semantics", {}),
    }


def verify_bundle(bundle: str | Path) -> dict:
    bundle = Path(bundle).resolve()
    ready_path = bundle / "ready.json"
    if not ready_path.is_file():
        raise ValueError("Dataset bundle is not annotation; re-prepare the dataset")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("annotation_schema") != "annotation" or ready.get("schema_version") != "dataset-bundle":
        raise ValueError("Dataset bundle is not annotation; re-prepare the dataset")
    access = ready["access"]
    for key in ("cases_path", "asset_manifest_path", "profile_path", "dataset_card_path"):
        if not Path(access[key]).is_file():
            raise ValueError(f"Missing bundle artifact: {key}")
    checks = {
        "annotation_digest": access["cases_path"],
        "asset_manifest_digest": access["asset_manifest_path"],
        "profile_digest": access["profile_path"],
    }
    for name in ("presentation", "split", "sampling"):
        path_key = f"{name}_manifest_path"
        digest_key = f"{name}_manifest_digest"
        if path_key in access or digest_key in access:
            if not access.get(path_key) or not access.get(digest_key):
                raise ValueError(f"Manifest path/digest pair is incomplete: {name}")
            checks[digest_key] = access[path_key]
    for digest_key, path in checks.items():
        if not Path(path).is_file():
            raise ValueError(f"Missing bundle artifact: {digest_key}")
        if digest_file(path) != access[digest_key]:
            raise ValueError(f"bundle digest mismatch: {digest_key}")
    for name, model in (
        ("presentation", PresentationRecord),
        ("split", SplitManifest),
        ("sampling", SamplingManifest),
    ):
        path = access.get(f"{name}_manifest_path")
        if path:
            load_manifest(path, model)
    card = read_card(access["dataset_card_path"])
    if card["annotation_digest"] != access["annotation_digest"] or card["asset_manifest_digest"] != access["asset_manifest_digest"] or card["profile_digest"] != access["profile_digest"]:
        raise ValueError("Dataset Card facts do not match bundle")
    return {"ready": ready, "card": card}

