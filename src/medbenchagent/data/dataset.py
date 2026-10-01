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
    # Kept as a diagnostic helper for callers that inspect the historical dataset
    # index; new preparation tasks resolve annotation.v5 explicitly.
    return Path(__file__).parents[1] / "schemas" / "annotation.v4.schema.json"




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


def prepare_annotation_dataset(annotation_path, cache_root, *, revision="1", refresh=False, scratch_root=None):
    annotation_path = Path(annotation_path)
    if annotation_path.is_file():
        with annotation_path.open(encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip()), "")
        if first:
            try:
                from medbenchagent.data.v5_models import CaseRecord
                CaseRecord.model_validate_json(first)
            except Exception:
                pass
            else:
                from medbenchagent.data.case_query import CaseQuery
                from medbenchagent.data.sources import digest_file
                manifest = annotation_path.parent / "asset_manifest.jsonl"
                query = CaseQuery(annotation_path, manifest)
                profile = query.profile().model_dump()
                profile_path = annotation_path.parent / "dataset_profile.json"
                profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                return profile, {
                    "schema_version": "annotation.v5",
                    "annotation_path": str(annotation_path.resolve()),
                    "asset_manifest_path": str(manifest.resolve()),
                    "profile_path": str(profile_path.resolve()),
                    "annotation_digest": digest_file(annotation_path),
                    "asset_manifest_digest": digest_file(manifest),
                    "profile_digest": digest_file(profile_path),
                    "revision": revision,
                }
    from medbenchagent.data.index import prepare_dataset
    return prepare_dataset(annotation_path, cache_root, revision=revision, refresh=refresh,
                           scratch_root=scratch_root)
