#!/usr/bin/env python3
"""Validate normalized annotation JSONL and optional converted-image references."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REQUIRED_FIELDS = (
    "image_id",
    "image_path",
    "patient_id",
    "case_id",
    "study_id",
    "annotation",
)


class ValidationError(ValueError):
    """Raised when an input violates the normalization contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"Required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _normalized_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string.")
    rendered = value.strip().replace("\\", "/")
    windows_path = PureWindowsPath(rendered)
    posix_path = PurePosixPath(rendered)
    if windows_path.is_absolute() or posix_path.is_absolute() or windows_path.drive:
        raise ValidationError(f"{field} must be relative: {value}")
    if ".." in posix_path.parts:
        raise ValidationError(f"{field} must not traverse parent directories: {value}")
    return posix_path.as_posix()


def verify_schema_parity(repo_root: Path | None = None) -> None:
    script_path = Path(__file__).resolve()
    skill_schema = script_path.parent.parent / "schema.json"
    root = repo_root.resolve() if repo_root else script_path.parents[4]
    project_schema = root / "src" / "medbenchagent" / "schemas" / "annotation.v1.schema.json"
    if _read_json(skill_schema) != _read_json(project_schema):
        raise ValidationError(
            f"Skill schema does not match the project contract: {skill_schema} != {project_schema}"
        )


def load_annotation_records(annotation_path: Path) -> list[dict[str, Any]]:
    if annotation_path.name != "annotation.jsonl":
        raise ValidationError("Input file must be named annotation.jsonl.")
    if not annotation_path.is_file():
        raise ValidationError(f"annotation.jsonl does not exist: {annotation_path}")

    records: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    with annotation_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValidationError(f"Line {line_number}: blank lines are not allowed.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Line {line_number}: invalid JSON.") from exc
            if not isinstance(record, dict):
                raise ValidationError(f"Line {line_number}: record must be a JSON object.")
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValidationError(
                    f"Line {line_number}: missing required fields: {', '.join(missing)}."
                )
            image_id = record["image_id"]
            if not isinstance(image_id, str) or not image_id.strip():
                raise ValidationError(f"Line {line_number}: image_id must be a non-empty string.")
            if image_id in image_ids:
                raise ValidationError(f"Line {line_number}: duplicate image_id: {image_id}.")
            image_ids.add(image_id)
            record["image_path"] = _normalized_relative_path(
                record["image_path"], field=f"Line {line_number} image_path"
            )
            for field in ("patient_id", "case_id", "study_id"):
                value = record[field]
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValidationError(
                        f"Line {line_number}: {field} must be a non-empty string or null."
                    )
            if not isinstance(record["annotation"], dict):
                raise ValidationError(f"Line {line_number}: annotation must be an object.")
            records.append(record)
    if not records:
        raise ValidationError("annotation.jsonl contains no records.")
    return records


def load_converted_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise ValidationError(f"Image conversion manifest does not exist: {manifest_path}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValidationError(f"Manifest line {line_number}: blank lines are not allowed.")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"Manifest line {line_number}: invalid JSON.") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"Manifest line {line_number}: row must be an object.")
            if row.get("status") != "converted":
                continue
            source_path = _normalized_relative_path(
                row.get("source_relative_path"),
                field=f"Manifest line {line_number} source_relative_path",
            )
            png_path = _normalized_relative_path(
                row.get("png_relative_path"),
                field=f"Manifest line {line_number} png_relative_path",
            )
            if PurePosixPath(png_path).suffix.lower() != ".png":
                raise ValidationError(
                    f"Manifest line {line_number}: png_relative_path must end in .png."
                )
            source_filename = row.get("source_filename")
            if not isinstance(source_filename, str) or not source_filename.strip():
                source_filename = PurePosixPath(source_path).name
            rows.append(
                {
                    "source_relative_path": source_path,
                    "source_filename": source_filename,
                    "source_stem": PurePosixPath(source_filename).stem,
                    "png_relative_path": png_path,
                }
            )
    if not rows:
        raise ValidationError("Image conversion manifest contains no converted rows.")
    return rows


def _unique_index(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field]).casefold()].append(row)
    duplicate_keys = sorted(key for key, values in grouped.items() if len(values) > 1)
    if duplicate_keys:
        preview = ", ".join(duplicate_keys[:5])
        raise ValidationError(f"Manifest has ambiguous {field} values: {preview}")
    return {key: values[0] for key, values in grouped.items()}


def validate_manifest_references(
    records: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]
) -> None:
    png_index = _unique_index(manifest_rows, "png_relative_path")
    source_path_index = _unique_index(manifest_rows, "source_relative_path")
    filename_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stem_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        filename_index[str(row["source_filename"]).casefold()].append(row)
        stem_index[str(row["source_stem"]).casefold()].append(row)

    for record in records:
        image_path = str(record["image_path"])
        key = image_path.casefold()
        if key in png_index:
            continue
        source_match = source_path_index.get(key)
        if source_match is None:
            filename_matches = filename_index.get(PurePosixPath(image_path).name.casefold(), [])
            if len(filename_matches) == 1:
                source_match = filename_matches[0]
            elif len(filename_matches) > 1:
                raise ValidationError(f"Ambiguous source filename in manifest: {image_path}")
        if source_match is None:
            stem_matches = stem_index.get(PurePosixPath(image_path).stem.casefold(), [])
            if len(stem_matches) == 1:
                source_match = stem_matches[0]
            elif len(stem_matches) > 1:
                raise ValidationError(f"Ambiguous source stem in manifest: {image_path}")
        if source_match is not None:
            raise ValidationError(
                f"image_path still references the source image {image_path}; expected "
                f"{source_match['png_relative_path']}"
            )
        raise ValidationError(f"image_path is missing from the converted manifest: {image_path}")


def validate_image_files(records: list[dict[str, Any]], image_root: Path) -> None:
    root = image_root.resolve()
    if not root.is_dir():
        raise ValidationError(f"Image root is not a directory: {root}")
    for record in records:
        relative = PurePosixPath(str(record["image_path"]))
        if relative.suffix.lower() != ".png":
            raise ValidationError(f"Mapped image is not a PNG: {relative}")
        resolved = root.joinpath(*relative.parts).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValidationError(f"Image path escapes image root: {relative}") from exc
        if not resolved.is_file():
            raise ValidationError(f"Mapped PNG does not exist: {resolved}")


def validate(
    annotation_path: Path,
    *,
    image_manifest: Path | None = None,
    image_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    verify_schema_parity(repo_root)
    records = load_annotation_records(annotation_path.resolve())
    manifest_rows: list[dict[str, Any]] = []
    if image_manifest is not None:
        manifest_rows = load_converted_manifest(image_manifest.resolve())
        validate_manifest_references(records, manifest_rows)
    if image_root is not None:
        validate_image_files(records, image_root)
    return {
        "annotation_path": str(annotation_path.resolve()),
        "record_count": len(records),
        "manifest_converted_count": len(manifest_rows) if image_manifest else None,
        "image_files_checked": len(records) if image_root else 0,
        "valid": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = validate(
            args.annotation,
            image_manifest=args.image_manifest,
            image_root=args.image_root,
        )
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
