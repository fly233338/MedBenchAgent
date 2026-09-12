"""Safe accessors for per-record generation context."""

from __future__ import annotations

import copy
import math
from typing import Any

_MISSING = object()


def _merge_projection(left: object, right: object) -> object:
    if isinstance(left, dict) and isinstance(right, dict):
        merged = copy.deepcopy(left)
        for key, value in right.items():
            merged[key] = (
                _merge_projection(merged[key], value) if key in merged else copy.deepcopy(value)
            )
        return merged
    if isinstance(left, list) and isinstance(right, list):
        if all(isinstance(item, dict) and "_source_index" in item for item in left + right):
            by_index = {int(item["_source_index"]): copy.deepcopy(item) for item in left}
            for item in right:
                index = int(item["_source_index"])
                by_index[index] = (
                    _merge_projection(by_index[index], item)
                    if index in by_index
                    else copy.deepcopy(item)
                )
            return [by_index[index] for index in sorted(by_index)]
    return copy.deepcopy(right)


def _project_value(value: object, segments: list[str]) -> object:
    if not segments:
        return copy.deepcopy(value)
    if not isinstance(value, dict):
        return _MISSING
    segment = segments[0]
    is_array = segment.endswith("[*]")
    key = segment.removesuffix("[*]")
    if key not in value:
        return _MISSING
    nested = value[key]
    if is_array:
        if not isinstance(nested, list):
            return _MISSING
        if len(segments) == 1:
            return {key: copy.deepcopy(nested)}
        projected_items: list[dict[str, Any]] = []
        for index, item in enumerate(nested):
            projected = _project_value(item, segments[1:])
            if projected is _MISSING:
                continue
            if isinstance(projected, dict):
                projected_items.append({"_source_index": index, **projected})
            else:
                projected_items.append({"_source_index": index, "value": projected})
        return {key: projected_items} if projected_items else _MISSING
    projected = _project_value(nested, segments[1:])
    return {key: projected} if projected is not _MISSING else _MISSING


def is_supporting_field_path(path: object) -> bool:
    """Return whether a path belongs to one task-supporting v2 scope."""
    return isinstance(path, str) and (
        path in {"image.modality", "image.anatomy"}
        or any(
            path.startswith(prefix)
            for prefix in (
                "patient.annotation.",
                "case.annotation.",
                "study.annotation.",
                "image.annotation.",
            )
        )
    )


def project_supporting_record(
    record: dict[str, Any], supporting_fields: list[str]
) -> dict[str, Any]:
    """Project approved v2 record paths while retaining array source indices."""
    projected: object = {}
    for path in supporting_fields:
        if not is_supporting_field_path(path):
            raise ValueError(f"Unsupported field path: {path}")
        value = _project_value(record, path.split("."))
        if value is not _MISSING:
            projected = _merge_projection(projected, value)
    return projected if isinstance(projected, dict) else {}


def has_meaningful_value(value: object) -> bool:
    """Return whether a projected value contains non-empty source data."""
    if value is None or value == "":
        return False
    if isinstance(value, dict):
        return any(
            key != "_source_index" and has_meaningful_value(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_meaningful_value(item) for item in value)
    return True


def record_values(candidate: Any) -> dict[str, object]:
    record = getattr(candidate, "record", {})
    return record if isinstance(record, dict) else {}


def image_modality(candidate: Any) -> str:
    image = record_values(candidate).get("image")
    modality = image.get("modality") if isinstance(image, dict) else None
    return modality if isinstance(modality, str) else ""


def evidence_value(record: dict[str, object], path: str, indices: list[int]) -> object:
    """Resolve one scalar v2 record value using source indices for wildcard segments."""
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


def _evidence_candidates(record: dict[str, object], path: str) -> list[dict[str, object]]:
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


def scalar_evidence_candidates(record: dict[str, object], path: str) -> list[dict[str, object]]:
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
            isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
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
