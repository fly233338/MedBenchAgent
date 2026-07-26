"""Safe accessors for per-record generation context."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def annotation_values(candidate: Any) -> dict[str, object]:
    record = getattr(candidate, "record", {})
    annotation = record.get("annotation") if isinstance(record, dict) else None
    return annotation if isinstance(annotation, dict) else {}


def annotation_modality(candidate: Any) -> str:
    modality = annotation_values(candidate).get("modality")
    return modality if isinstance(modality, str) else ""


def evidence_value(annotation: dict[str, object], path: str, indices: list[int]) -> object:
    """Resolve one scalar annotation value using source indices for wildcard segments."""
    if not path.startswith("annotation."):
        raise ValueError("Evidence path must start with annotation.")
    segments = path.split(".")[1:]
    if len(indices) != sum(segment.endswith("[*]") for segment in segments):
        raise ValueError("Evidence indices must match the path wildcards.")
    current: object = annotation
    index_cursor = 0
    for segment in segments:
        wildcard = segment.endswith("[*]")
        key = segment.removesuffix("[*]")
        if not isinstance(current, dict) or key not in current:
            raise ValueError("Evidence path does not exist in the annotation.")
        current = current[key]
        if wildcard:
            index = indices[index_cursor]
            index_cursor += 1
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or not isinstance(current, list)
                or index >= len(current)
            ):
                raise ValueError("Evidence index is invalid for the annotation path.")
            current = current[index]
    if (
        current is None
        or current == ""
        or isinstance(current, (dict, list))
        or isinstance(current, float)
        and not math.isfinite(current)
    ):
        raise ValueError("Evidence must resolve to one non-empty JSON scalar.")
    return current


def evidence_sha256(path: str, indices: list[int], value: object) -> str:
    payload = json.dumps(
        {"path": path, "indices": indices, "value": value},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_evidence(
    annotation: dict[str, object],
    supporting_fields: object,
    evidence: object,
) -> dict[str, object]:
    """Validate an LLM-copied witness without normalizing its source value."""
    if not isinstance(evidence, dict) or set(evidence) != {"path", "indices", "value"}:
        raise ValueError("Evidence must contain only path, indices, and value.")
    path, indices, value = evidence["path"], evidence["indices"], evidence["value"]
    if (
        not isinstance(path, str)
        or not isinstance(supporting_fields, list)
        or path not in supporting_fields
    ):
        raise ValueError("Evidence path must be one exact task supporting field.")
    if not isinstance(indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) and index >= 0
        for index in indices
    ):
        raise ValueError("Evidence indices must be non-negative integers.")
    actual = evidence_value(annotation, path, indices)
    if type(actual) is not type(value) or actual != value:
        raise ValueError("Evidence value must exactly copy the indexed annotation value.")
    return {"path": path, "indices": list(indices), "value": value}


def evidence_provenance(evidence: dict[str, object]) -> dict[str, object]:
    path = str(evidence["path"])
    indices = list(evidence["indices"]) if isinstance(evidence["indices"], list) else []
    return {
        "path": path,
        "indices": indices,
        "evidence_sha256": evidence_sha256(path, indices, evidence["value"]),
    }
