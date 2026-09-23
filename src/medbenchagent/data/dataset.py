"""Validated image-level annotation JSONL storage and deterministic profiling."""

from __future__ import annotations

import json
from pathlib import Path
from medbenchagent.data.validation import AnnotationValidationError as AnnotationValidationError



PROFILE_SCHEMA_VERSION = "task-capability-profile.v4"
CACHE_FORMAT_VERSION = "annotation-cache.v9"

TOP_VALUE_EXCLUDED_PREFIXES = (
    "image.annotation.dicom_metadata.",
)
TOP_VALUE_EXCLUDED_SEGMENTS = {
    "split",
    "fold",
    "partition",
}
IDENTIFIER_SUFFIXES = (
    "_id",
    "_uid",
    "_uuid",
)
FILE_REFERENCE_SEGMENTS = {
    "path",
    "paths",
    "filename",
    "filenames",
}
FILE_REFERENCE_SUFFIXES = (
    "_path",
    "_paths",
    "_filename",
    "_filenames",
)


def annotation_schema_path() -> Path:
    return Path(__file__).parents[1] / "schemas" / "annotation.v3.schema.json"




def _value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _is_non_empty(value: object) -> bool:
    return value is not None and value not in ("", [], {})


def _should_include_top_values(path: str) -> bool:
    """Expose bounded semantic examples without identifiers, files, metadata, or coordinates."""
    if path.startswith(TOP_VALUE_EXCLUDED_PREFIXES):
        return False
    segments = [segment.removesuffix("[*]") for segment in path.split(".")]
    if any(segment in TOP_VALUE_EXCLUDED_SEGMENTS for segment in segments):
        return False
    if any(segment in {"id", "uid", "uuid"} or segment.endswith(IDENTIFIER_SUFFIXES) for segment in segments):
        return False
    if "bbox" in segments:
        return False
    if any(
        segment in FILE_REFERENCE_SEGMENTS or segment.endswith(FILE_REFERENCE_SUFFIXES)
        for segment in segments
    ):
        return False
    return True


def _profile_top_value(value: str, max_chars: int) -> object:
    rendered = json.loads(value)
    if isinstance(rendered, str) and len(rendered) > max_chars:
        return f"{rendered[: max_chars - 1]}…" if max_chars > 1 else "…"
    return rendered


def build_dataset_profile(records):
    from medbenchagent.data.index import build_profile
    return build_profile(records)


def prepare_annotation_dataset(annotation_path, cache_root, *, revision="1", refresh=False):
    from medbenchagent.data.index import prepare_dataset
    return prepare_dataset(annotation_path, cache_root, revision=revision, refresh=refresh)
