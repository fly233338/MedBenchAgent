"""Catalog and Card contracts for immutable annotation bundles."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.index import prepare_dataset
from medbenchagent.runtime.run_artifacts import atomic_write_json


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


def verify_bundle(bundle):
    """Verify committed content and reconstruct cache/Card outside the dataset revision."""
    from medbenchagent.preparation.validation import bundle_digest
    bundle = Path(bundle).resolve()
    if (bundle / "current.json").is_file():
        receipt = json.loads((bundle / "current.json").read_text(encoding="utf-8"))
        revision = Path(receipt["revision_path"]).resolve()
        if not revision.is_relative_to(bundle / "revisions"):
            raise ValueError("Current points outside committed revisions")
        root = bundle
    else:
        revision, root = bundle, bundle.parent.parent
        receipt_path = root / ".commits" / f"{revision.name}.json"
        if not receipt_path.is_file():
            raise ValueError("Dataset revision is not committed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if bundle_digest(revision) != receipt["digest"]:
        raise ValueError("Dataset revision digest mismatch")
    metadata = json.loads((revision / "metadata.json").read_text(encoding="utf-8"))
    cache = root / ".derived" / receipt["revision"]
    profile, access = prepare_dataset(revision / "cases.jsonl", revision / "image_manifest.jsonl", cache,
                                      revision=receipt["revision"])
    card = {"schema_version": "dataset-card", "dataset_id": receipt["dataset_id"],
            "name": metadata.get("name", receipt["dataset_id"]), "description": metadata.get("description", "annotation dataset"),
            "annotation_digest": access["annotation_digest"], "image_manifest_digest": access["image_manifest_digest"],
            "profile_digest": access["profile_digest"], "case_count": profile["case_count"], "patient_count": profile["patient_count"],
            "semantics": metadata.get("field_semantics", {})}
    card_path = cache / "dataset_card.json"
    if not card_path.is_file() or json.loads(card_path.read_text(encoding="utf-8")) != card:
        atomic_write_json(card_path, card)
    access.update(dataset_card_path=str(card_path), bundle_root=str(revision), mapping_path=str(revision / "metadata.json"))
    return {"ready": {"revision": receipt["revision"], "access": access}, "card": card}
