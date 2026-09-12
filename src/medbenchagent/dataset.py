"""Validated image-level annotation JSONL storage and deterministic profiling."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.settings import get_settings

RECORD_ENTITIES = ("patient", "case", "study", "image")
ENTITY_FIELDS = {
    "patient": {"patient_id", "annotation"},
    "case": {"case_id", "annotation"},
    "study": {"study_id", "annotation"},
    "image": {"image_id", "image_path", "modality", "anatomy", "annotation"},
}
PROFILE_SCHEMA_VERSION = "task-capability-profile.v3"
CACHE_FORMAT_VERSION = "annotation-cache.v7"

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


class AnnotationValidationError(ValueError):
    """An annotation JSONL record does not meet the stable input contract."""


def annotation_schema_path() -> Path:
    return Path(__file__).parent / "schemas" / "annotation.v2.schema.json"




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


def _validate_optional_id(value: object, field: str, line_number: int) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise AnnotationValidationError(f"Line {line_number}: {field} must be a non-empty string or null.")


def load_annotation_jsonl(annotation_path: str | Path) -> list[dict[str, Any]]:
    """Validate entity-scoped image records without accessing image files."""
    path = Path(annotation_path).expanduser().resolve()
    if path.name != "annotation.jsonl":
        raise AnnotationValidationError("Input file must be named annotation.jsonl.")
    if not path.is_file():
        raise AnnotationValidationError(f"annotation.jsonl does not exist: {path}")

    records: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    image_paths: set[str] = set()
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
            if set(record) != set(RECORD_ENTITIES):
                raise AnnotationValidationError(
                    f"Line {line_number}: record must contain only patient, case, study, and image."
                )
            for entity_name in RECORD_ENTITIES:
                entity = record[entity_name]
                if not isinstance(entity, dict) or set(entity) != ENTITY_FIELDS[entity_name]:
                    expected = ", ".join(sorted(ENTITY_FIELDS[entity_name]))
                    raise AnnotationValidationError(
                        f"Line {line_number}: {entity_name} must contain only {expected}."
                    )
                if not isinstance(entity["annotation"], dict):
                    raise AnnotationValidationError(
                        f"Line {line_number}: {entity_name}.annotation must be a JSON object."
                    )

            patient = record["patient"]
            case = record["case"]
            study = record["study"]
            image = record["image"]
            _validate_optional_id(patient["patient_id"], "patient.patient_id", line_number)
            _validate_optional_id(case["case_id"], "case.case_id", line_number)
            _validate_optional_id(study["study_id"], "study.study_id", line_number)

            image_id = image["image_id"]
            if not isinstance(image_id, str) or not image_id.strip():
                raise AnnotationValidationError(
                    f"Line {line_number}: image.image_id must be a non-empty string."
                )
            if image_id in image_ids:
                raise AnnotationValidationError(
                    f"Line {line_number}: duplicate image.image_id: {image_id}."
                )
            image_ids.add(image_id)
            image_path = image["image_path"]
            if not isinstance(image_path, str) or not image_path.strip():
                raise AnnotationValidationError(
                    f"Line {line_number}: image.image_path must be a non-empty string."
                )
            if image_path in image_paths:
                raise AnnotationValidationError(
                    f"Line {line_number}: duplicate image.image_path: {image_path}."
                )
            image_paths.add(image_path)
            for field in ("modality", "anatomy"):
                value = image[field]
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise AnnotationValidationError(
                        f"Line {line_number}: image.{field} must be a non-empty string or null."
                    )
            records.append({**record, "_source_row": line_number})
    if not records:
        raise AnnotationValidationError("annotation.jsonl contains no records.")
    return records


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


def _is_null_only_bbox_parent(
    path: str, item: dict[str, Any], structures: dict[str, dict[str, Any]]
) -> bool:
    structure = structures.get(path)
    return (
        path.endswith(".bbox")
        and isinstance(structure, dict)
        and structure.get("kind") == "array"
        and item["types"] == {"null"}
        and not item["non_empty_images"]
    )


def _profile_top_value(value: str, max_chars: int) -> object:
    rendered = json.loads(value)
    if isinstance(rendered, str) and len(rendered) > max_chars:
        return f"{rendered[: max_chars - 1]}…" if max_chars > 1 else "…"
    return rendered


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


def build_dataset_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize task-supporting entity paths without inferring medical semantics."""
    fields_stats: dict[str, dict[str, Any]] = {}
    structures_stats: dict[str, dict[str, Any]] = {}
    for image_index, record in enumerate(records):
        for entity_name in ("patient", "case", "study"):
            _collect_profile_values(
                record[entity_name]["annotation"],
                f"{entity_name}.annotation",
                image_index,
                fields_stats,
                structures_stats,
            )
        image = record["image"]
        for field in ("modality", "anatomy"):
            _collect_profile_values(
                image[field],
                f"image.{field}",
                image_index,
                fields_stats,
                structures_stats,
            )
        _collect_profile_values(
            image["annotation"],
            "image.annotation",
            image_index,
            fields_stats,
            structures_stats,
        )

    fields: dict[str, Any] = {}
    profile_settings = get_settings().dataset_profile
    for path in sorted(fields_stats):
        item = fields_stats[path]
        if _is_null_only_bbox_parent(path, item, structures_stats):
            continue
        value_types = item["types"]
        distinct_value_count = len(item["values"])
        field = {
            "path": path,
            "types": sorted(value_types),
            "non_empty_images": len(item["non_empty_images"]),
            "distinct_value_count": distinct_value_count,
        }
        if _should_include_top_values(path):
            ordered_values = sorted(
                item["values"].items(), key=lambda pair: (-pair[1], pair[0])
            )
            top_values = ordered_values[: profile_settings.max_top_values]
            field["top_values"] = [
                _profile_top_value(value, profile_settings.max_top_value_chars)
                for value, _count in top_values
            ]
            if ordered_values:
                total_count = sum(item["values"].values())
                least_value, least_count = min(
                    item["values"].items(), key=lambda pair: (pair[1], pair[0])
                )
                field["value_distribution"] = {
                    "total_count": total_count,
                    "top": [
                        {
                            "value": _profile_top_value(
                                value, profile_settings.max_top_value_chars
                            ),
                            "count": count,
                            "proportion": count / total_count,
                        }
                        for value, count in top_values
                    ],
                    "least_frequent": {
                        "value": _profile_top_value(
                            least_value, profile_settings.max_top_value_chars
                        ),
                        "count": least_count,
                        "proportion": least_count / total_count,
                    },
                    "truncated": distinct_value_count > len(top_values),
                }
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
                record JSON NOT NULL,
                _source_row BIGINT NOT NULL
            )"""
        )
        connection.executemany(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, CAST(? AS JSON), ?)",
            [
                (
                    record["image"]["image_id"],
                    record["image"]["image_path"],
                    record["patient"]["patient_id"],
                    record["case"]["case_id"],
                    record["study"]["study_id"],
                    json.dumps(
                        {entity: record[entity] for entity in RECORD_ENTITIES},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    record["_source_row"],
                )
                for record in records
            ],
        )
    finally:
        connection.close()


def prepare_annotation_dataset(annotation_path: str | Path, cache_root: str | Path, *, revision: str = "1", refresh: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate JSONL and build or reuse its explicitly versioned DuckDB/profile cache."""
    source_path = Path(annotation_path).expanduser().resolve()
    if source_path.name != "annotation.jsonl":
        raise AnnotationValidationError("Input file must be named annotation.jsonl.")
    if not source_path.is_file():
        raise AnnotationValidationError(f"annotation.jsonl does not exist: {source_path}")
    if not revision or not revision.isalnum():
        raise ValueError("Cache revision must be alphanumeric.")
    source_key = source_path.as_posix().replace(":", "").lstrip("/")
    profile_settings = get_settings().dataset_profile
    profile_key = f"top-{profile_settings.max_top_values}-chars-{profile_settings.max_top_value_chars}"
    cache_dir = Path(cache_root) / CACHE_FORMAT_VERSION / profile_key / source_key / revision
    database_path = cache_dir / "dataset.duckdb"
    profile_path = cache_dir / "dataset_profile.json"
    manifest_path = cache_dir / "dataset_access_manifest.json"
    if not refresh and database_path.is_file() and profile_path.is_file() and manifest_path.is_file():
        return (
            json.loads(profile_path.read_text(encoding="utf-8")),
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )

    records = load_annotation_jsonl(source_path)
    profile = build_dataset_profile(records)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    if database_path.exists():
        database_path.unlink()
    _write_database(records, database_path)
    manifest = {
        "annotation_path": str(source_path),
        "revision": revision,
        "cache_format_version": CACHE_FORMAT_VERSION,
        "duckdb_path": str(database_path),
        "table": "images",
        "profile_path": str(profile_path),
    }
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile, manifest
