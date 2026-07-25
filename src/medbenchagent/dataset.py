"""Validated image-level annotation JSONL storage and deterministic profiling."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.settings import get_settings

REQUIRED_RECORD_FIELDS = (
    "image_id",
    "image_path",
    "patient_id",
    "case_id",
    "study_id",
    "annotation",
)
PROFILE_SCHEMA_VERSION = "task-capability-profile.v1"
CACHE_FORMAT_VERSION = "annotation-cache.v3"

TOP_VALUE_EXCLUDED_PREFIXES = (
    "annotation.dicom_metadata.",
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


class AnnotationValidationError(ValueError):
    """An annotation JSONL record does not meet the stable input contract."""


def annotation_schema_path() -> Path:
    return Path(__file__).parent / "schemas" / "annotation.v1.schema.json"


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def _validate_optional_id(record: dict[str, Any], field: str, line_number: int) -> None:
    value = record[field]
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise AnnotationValidationError(f"Line {line_number}: {field} must be a non-empty string or null.")


def load_annotation_jsonl(annotation_path: str | Path) -> list[dict[str, Any]]:
    """Validate annotation records without accessing image files."""
    path = Path(annotation_path).expanduser().resolve()
    if path.name != "annotation.jsonl":
        raise AnnotationValidationError("Input file must be named annotation.jsonl.")
    if not path.is_file():
        raise AnnotationValidationError(f"annotation.jsonl does not exist: {path}")

    records: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise AnnotationValidationError(f"Line {line_number}: blank lines are not allowed.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(f"Line {line_number}: invalid JSON.") from exc
            if not isinstance(record, dict):
                raise AnnotationValidationError(f"Line {line_number}: record must be a JSON object.")
            missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
            if missing:
                raise AnnotationValidationError(f"Line {line_number}: missing required fields: {', '.join(missing)}.")
            image_id = record["image_id"]
            if not isinstance(image_id, str) or not image_id.strip():
                raise AnnotationValidationError(f"Line {line_number}: image_id must be a non-empty string.")
            if image_id in image_ids:
                raise AnnotationValidationError(f"Line {line_number}: duplicate image_id: {image_id}.")
            image_ids.add(image_id)
            image_path = record["image_path"]
            if not isinstance(image_path, str) or not image_path.strip():
                raise AnnotationValidationError(f"Line {line_number}: image_path must be a non-empty string.")
            for field in ("patient_id", "case_id", "study_id"):
                _validate_optional_id(record, field, line_number)
            if not isinstance(record["annotation"], dict):
                raise AnnotationValidationError(f"Line {line_number}: annotation must be a JSON object.")
            records.append({**record, "_source_row": line_number})
    if not records:
        raise AnnotationValidationError("annotation.jsonl contains no records.")
    return records


def _should_include_top_values(path: str, value_types: set[str], distinct_value_count: int) -> bool:
    """Keep categorical values useful for task support without exposing sensitive/raw values."""
    if path.startswith(TOP_VALUE_EXCLUDED_PREFIXES):
        return False
    segments = [segment.removesuffix("[*]") for segment in path.split(".")]
    if any(segment in TOP_VALUE_EXCLUDED_SEGMENTS for segment in segments):
        return False
    if any(segment in {"id", "uid", "uuid"} or segment.endswith(IDENTIFIER_SUFFIXES) for segment in segments):
        return False
    if "boolean" in value_types and value_types <= {"boolean", "null"}:
        return True
    return (
        "string" in value_types
        and distinct_value_count <= get_settings().dataset_profile.max_categorical_distinct_values
    )


def _collect_profile_values(
    value: object,
    path: str,
    image_index: int,
    fields: dict[str, dict[str, Any]],
    structures: dict[str, dict[str, Any]],
) -> None:
    if isinstance(value, dict):
        item = structures.setdefault(path, {"path": path, "kind": "object", "children": set()})
        for key, nested_value in value.items():
            child_path = f"{path}.{key}"
            item["children"].add(child_path)
            _collect_profile_values(nested_value, child_path, image_index, fields, structures)
    elif isinstance(value, list):
        item_path = f"{path}[*]"
        item = structures.setdefault(
            path,
            {"path": path, "kind": "array", "item_path": item_path, "children": set()},
        )
        for nested_value in value:
            if isinstance(nested_value, dict):
                for key in nested_value:
                    child_path = f"{item_path}.{key}"
                    item["children"].add(child_path)
                    _collect_profile_values(nested_value[key], child_path, image_index, fields, structures)
            elif isinstance(nested_value, list):
                item["children"].add(f"{item_path}[*]")
                _collect_profile_values(nested_value, item_path, image_index, fields, structures)
            else:
                _collect_profile_values(nested_value, item_path, image_index, fields, structures)
    else:
        item = fields.setdefault(
            path,
            {
                "types": set(),
                "non_empty_images": set(),
                "values": Counter(),
            },
        )
        item["types"].add(_value_type(value))
        if _is_non_empty(value):
            item["non_empty_images"].add(image_index)
            item["values"][_canonical_value(value)] += 1


def build_dataset_profile(records: list[dict[str, Any]], content_hash: str) -> dict[str, Any]:
    """Summarize annotation paths deterministically without inferring medical semantics."""
    fields_stats: dict[str, dict[str, Any]] = {}
    structures_stats: dict[str, dict[str, Any]] = {}
    for image_index, record in enumerate(records):
        _collect_profile_values(record["annotation"], "annotation", image_index, fields_stats, structures_stats)

    fields: dict[str, Any] = {}
    for path in sorted(fields_stats):
        item = fields_stats[path]
        value_types = item["types"]
        distinct_value_count = len(item["values"])
        field = {
            "path": path,
            "types": sorted(value_types),
            "non_empty_images": len(item["non_empty_images"]),
            "distinct_value_count": distinct_value_count,
        }
        if _should_include_top_values(path, value_types, distinct_value_count):
            field["top_values"] = [
                json.loads(value)
                for value, _count in sorted(item["values"].items(), key=lambda pair: (-pair[1], pair[0]))[
                    : get_settings().dataset_profile.max_top_values
                ]
            ]
        fields[path] = field

    structures = []
    for path in sorted(structures_stats):
        structure = structures_stats[path]
        rendered = {key: value for key, value in structure.items() if key != "children"}
        rendered["children"] = sorted(structure["children"])
        structures.append(rendered)
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "total_images": len(records),
        "structures": structures,
        "fields": fields,
    }


def _write_database(records: list[dict[str, Any]], database_path: Path) -> None:
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(
            """CREATE TABLE images (
                image_id VARCHAR PRIMARY KEY,
                image_path VARCHAR NOT NULL,
                patient_id VARCHAR,
                case_id VARCHAR,
                study_id VARCHAR,
                annotation JSON NOT NULL,
                _source_row BIGINT NOT NULL
            )"""
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, CAST(? AS JSON), ?)",
            [
                (
                    record["image_id"],
                    record["image_path"],
                    record["patient_id"],
                    record["case_id"],
                    record["study_id"],
                    json.dumps(record["annotation"], ensure_ascii=False, separators=(",", ":")),
                    record["_source_row"],
                )
                for record in records
            ],
        )
    finally:
        connection.close()


def prepare_annotation_dataset(annotation_path: str | Path, cache_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate JSONL and build or reuse its content-addressed DuckDB/profile cache."""
    source_path = Path(annotation_path).expanduser().resolve()
    if source_path.name != "annotation.jsonl":
        raise AnnotationValidationError("Input file must be named annotation.jsonl.")
    if not source_path.is_file():
        raise AnnotationValidationError(f"annotation.jsonl does not exist: {source_path}")
    content_hash = _content_hash(source_path)
    profile_settings = get_settings().dataset_profile
    profile_key = (
        f"top-{profile_settings.max_top_values}-"
        f"categorical-{profile_settings.max_categorical_distinct_values}"
    )
    cache_dir = Path(cache_root) / CACHE_FORMAT_VERSION / profile_key / content_hash
    database_path = cache_dir / "dataset.duckdb"
    profile_path = cache_dir / "dataset_profile.json"
    manifest_path = cache_dir / "dataset_access_manifest.json"
    if database_path.is_file() and profile_path.is_file() and manifest_path.is_file():
        return (
            json.loads(profile_path.read_text(encoding="utf-8")),
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )

    records = load_annotation_jsonl(source_path)
    profile = build_dataset_profile(records, content_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    _write_database(records, database_path)
    manifest = {
        "annotation_path": str(source_path),
        "annotation_sha256": content_hash,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "duckdb_path": str(database_path),
        "table": "images",
        "provenance_fields": ["_source_row"],
        "profile_path": str(profile_path),
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile, manifest
