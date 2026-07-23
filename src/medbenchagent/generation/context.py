"""Safe accessors for per-record generation context."""

from __future__ import annotations

from typing import Any


def annotation_values(candidate: Any) -> dict[str, object]:
    record = getattr(candidate, "record", {})
    annotation = record.get("annotation") if isinstance(record, dict) else None
    return annotation if isinstance(annotation, dict) else {}


def annotation_modality(candidate: Any) -> str:
    modality = annotation_values(candidate).get("modality")
    return modality if isinstance(modality, str) else ""
