"""Safe accessors for per-record generation context."""

from __future__ import annotations

from typing import Any


def annotation_metadata(candidate: Any) -> dict[str, object]:
    record = getattr(candidate, "record", {})
    annotation = record.get("annotation") if isinstance(record, dict) else None
    metadata = annotation.get("metadata") if isinstance(annotation, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def annotation_modality(candidate: Any) -> str:
    modality = annotation_metadata(candidate).get("modality")
    return modality if isinstance(modality, str) else ""
