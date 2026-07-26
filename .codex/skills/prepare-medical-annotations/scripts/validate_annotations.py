"""Validate canonical annotation records before image preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path.")
    rendered = value.strip().replace("\\", "/")
    windows = PureWindowsPath(rendered)
    posix = PurePosixPath(rendered)
    if (
        windows.is_absolute()
        or bool(windows.drive)
        or posix.is_absolute()
        or ".." in posix.parts
        or str(posix) == "."
    ):
        raise ValueError(f"{field} must be a non-escaping relative path: {value}")
    return posix.as_posix()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"{path}: line {line_number} is blank.")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number} is not JSON.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} must be an object.")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no records.")
    return rows


def validate(annotation_path: Path, schema_path: Path, source_root: Path) -> dict[str, object]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required_fields = schema.get("required")
    if not isinstance(required_fields, list) or not all(
        isinstance(field, str) for field in required_fields
    ):
        raise ValueError("Annotation schema has an invalid required-field contract.")
    required = set(required_fields)
    root = source_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Source-image root is not a directory: {root}")

    rows = _rows(annotation_path)
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not required <= set(row):
            raise ValueError(f"Annotation row {index} is missing canonical fields.")
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip() or image_id in image_ids:
            raise ValueError(f"Annotation row {index} has an invalid image_id.")
        if not isinstance(row.get("annotation"), dict):
            raise ValueError(f"Annotation row {index} has a non-object annotation.")
        for field in ("patient_id", "case_id", "study_id"):
            value = row.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Annotation row {index} has an invalid {field}.")
        relative = _relative(row.get("image_path"), f"annotation row {index} image_path")
        if relative in image_paths:
            raise ValueError(f"Annotation row {index} has a duplicate image_path.")
        resolved = root.joinpath(*PurePosixPath(relative).parts).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Annotation row {index} escapes the source-image root.") from exc
        if not resolved.is_file():
            raise ValueError(f"Annotation row {index} references a missing image: {relative}")
        image_ids.add(image_id)
        image_paths.add(relative)

    return {
        "records": len(rows),
        "referenced_files_checked": len(rows),
        "annotation_sha256": _sha256(annotation_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parents[4]
        / "src"
        / "medbenchagent"
        / "schemas"
        / "annotation.v1.schema.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.annotation, args.schema, args.source_image_root),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
