"""Build a deterministic image_id-to-PNG mapping from preparation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path.")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError(f"{field} must not be absolute or traverse directories: {value}")
    return path.as_posix()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_number} is not JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}: line {line_number} must be an object.")
        rows.append(row)
    return rows


def build(
    annotation_path: Path,
    conversion_path: Path,
    *,
    png_root_relative_to_manifest: PurePosixPath = PurePosixPath("."),
) -> list[dict[str, object]]:
    converted: dict[str, dict[str, Any]] = {}
    for row in _rows(conversion_path):
        source = _relative(row.get("source_relative_path"), "source_relative_path")
        if row.get("status") != "converted":
            continue
        if source in converted:
            raise ValueError(f"Conversion manifest has duplicate successful source path: {source}")
        converted[source] = row
    output = []
    image_ids: set[str] = set()
    sources: set[str] = set()
    for row in _rows(annotation_path):
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id or image_id in image_ids:
            raise ValueError("annotation image_id values must be unique non-empty strings.")
        source = _relative(row.get("image_path"), "annotation.image_path")
        if source in sources:
            raise ValueError(f"Annotation has duplicate image_path: {source}")
        conversion = converted.get(source)
        if conversion is None:
            raise ValueError(f"No successful PNG conversion for annotation image_path: {source}")
        png = _relative(conversion.get("png_relative_path"), "png_relative_path")
        width, height = conversion.get("width"), conversion.get("height")
        digest = conversion.get("output_sha256")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid PNG dimensions for: {source}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid PNG SHA-256 for: {source}")
        manifest_png = png_root_relative_to_manifest / png
        output.append({"image_id": image_id, "source_path": source, "png_path": manifest_png.as_posix(), "width": width,
                       "height": height, "sha256": digest, "conversion_status": "success",
                       "transform": {"spatial": "none"}})
        image_ids.add(image_id)
        sources.add(source)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-root", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_root = args.png_root.resolve() if args.png_root else args.output.parent.resolve()
    manifest_root = args.output.parent.resolve()
    try:
        png_root_relative = PurePosixPath(png_root.relative_to(manifest_root).as_posix())
    except ValueError as exc:
        raise ValueError("--png-root must be inside the image manifest directory.") from exc
    rows = build(
        args.annotation,
        args.conversion_manifest,
        png_root_relative_to_manifest=png_root_relative,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
