"""Safe accessors for per-record generation context."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def record_values(candidate: Any) -> dict[str, object]:
    record = getattr(candidate, "record", {})
    return record if isinstance(record, dict) else {}


def image_modality(candidate: Any) -> str:
    image = record_values(candidate).get("image")
    modality = image.get("modality") if isinstance(image, dict) else None
    return modality if isinstance(modality, str) else ""


def evidence_value(record: dict[str, object], path: str, indices: list[int]) -> object:
    """Resolve one scalar v2 record value using source indices for wildcard segments."""
    from medbenchagent.generation.engine import is_supporting_field_path

    if not is_supporting_field_path(path):
        raise ValueError("Evidence path must be one supported v2 entity field path.")
    segments = path.split(".")
    if len(indices) != sum(segment.endswith("[*]") for segment in segments):
        raise ValueError("Evidence indices must match the path wildcards.")
    current: object = record
    index_cursor = 0
    for segment in segments:
        wildcard = segment.endswith("[*]")
        key = segment.removesuffix("[*]")
        if not isinstance(current, dict) or key not in current:
            raise ValueError("Evidence path does not exist in the record.")
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


def _evidence_candidates(
    record: dict[str, object], path: str
) -> list[dict[str, object]]:
    from medbenchagent.generation.engine import is_supporting_field_path

    if not is_supporting_field_path(path):
        raise ValueError("Evidence path must be one supported v2 entity field path.")
    segments = path.split(".")
    index_sets: list[list[int]] = []

    def visit(current: object, remaining: list[str], indices: list[int]) -> None:
        if not remaining:
            index_sets.append(indices)
            return
        segment = remaining[0]
        wildcard = segment.endswith("[*]")
        key = segment.removesuffix("[*]")
        if not isinstance(current, dict) or key not in current:
            return
        nested = current[key]
        if wildcard:
            if not isinstance(nested, list):
                return
            for index, item in enumerate(nested):
                visit(item, remaining[1:], [*indices, index])
            return
        visit(nested, remaining[1:], indices)

    visit(record, segments, [])
    candidates = []
    for indices in index_sets:
        try:
            value = evidence_value(record, path, indices)
        except ValueError:
            continue
        candidates.append({"path": path, "indices": indices, "value": value})
    return candidates


def scalar_evidence_candidates(
    record: dict[str, object], path: str
) -> list[dict[str, object]]:
    """Return every non-empty scalar witness for one approved field path."""
    return _evidence_candidates(record, path)


def locate_scalar_evidence(
    record: dict[str, object], supporting_fields: object, expected: object
) -> dict[str, object]:
    """Locate one exact scalar source witness among approved supporting fields."""
    if not isinstance(supporting_fields, list):
        raise ValueError("Supporting fields must be a list.")
    matches = [
        evidence
        for path in supporting_fields
        if isinstance(path, str)
        for evidence in _evidence_candidates(record, path)
        if type(evidence["value"]) is type(expected) and evidence["value"] == expected
    ]
    if len(matches) != 1:
        raise ValueError("Answer must match exactly one supporting scalar value.")
    return matches[0]


def normalized_bbox(value: object) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(item)
            for item in value
        )
    ):
        return None
    bbox = [float(item) for item in value]
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def locate_bbox_evidence(
    record: dict[str, object], supporting_fields: object, expected: object
) -> list[dict[str, object]]:
    """Locate one xyxy box represented by four terminal wildcard scalar values."""
    target = normalized_bbox(expected)
    if target is None or not isinstance(supporting_fields, list):
        raise ValueError("A bbox witness requires one valid xyxy answer.")
    matches: list[list[dict[str, object]]] = []
    for path in supporting_fields:
        if not isinstance(path, str) or not path.endswith("[*]"):
            continue
        grouped: dict[tuple[int, ...], dict[int, dict[str, object]]] = {}
        for evidence in _evidence_candidates(record, path):
            indices = evidence["indices"]
            if not isinstance(indices, list) or not indices:
                continue
            grouped.setdefault(tuple(indices[:-1]), {})[indices[-1]] = evidence
        for coordinates in grouped.values():
            if set(coordinates) != {0, 1, 2, 3}:
                continue
            ordered = [coordinates[index] for index in range(4)]
            if normalized_bbox([evidence["value"] for evidence in ordered]) == target:
                matches.append(ordered)
    if len(matches) != 1:
        raise ValueError("Answer must match exactly one supporting bbox.")
    return matches[0]


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
    record: dict[str, object],
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
    actual = evidence_value(record, path, indices)
    if type(actual) is not type(value) or actual != value:
        raise ValueError("Evidence value must exactly copy the indexed annotation value.")
    return {"path": path, "indices": list(indices), "value": value}


def evidence_provenance(
    evidence: dict[str, object] | list[dict[str, object]],
) -> dict[str, object] | list[dict[str, object]]:
    def compact(item: dict[str, object]) -> dict[str, object]:
        path = str(item["path"])
        indices = list(item["indices"]) if isinstance(item["indices"], list) else []
        return {
            "path": path,
            "indices": indices,
            "evidence_sha256": evidence_sha256(path, indices, item["value"]),
        }

    if isinstance(evidence, list):
        return [compact(item) for item in evidence]
    return compact(evidence)
