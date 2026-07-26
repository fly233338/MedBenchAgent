"""Build and verify the exact image_id-to-PNG manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image

HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ALLOWED_MODES = {"L", "RGB"}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
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


def _inside(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*PurePosixPath(relative).parts).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"PNG path escapes its root: {relative}") from exc
    return resolved


def verify_png(row: dict[str, Any], png_root: Path) -> tuple[str, int, int, str]:
    relative = _relative(row.get("png_relative_path"), "png_relative_path")
    path = _inside(png_root, relative)
    digest = row.get("output_sha256")
    width, height, mode = row.get("width"), row.get("height"), row.get("output_mode")
    normalization = row.get("normalization")
    if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
        raise ValueError(f"Invalid PNG SHA-256 for: {relative}")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions for: {relative}")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported PNG mode for {relative}: {mode}")
    if not isinstance(normalization, str) or "resized_max_edge_" in normalization:
        raise ValueError(f"Spatially transformed PNG is not allowed: {relative}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"PNG does not exist: {relative}")
    if _sha256(path) != digest:
        raise ValueError(f"PNG hash mismatch: {relative}")
    try:
        with Image.open(path) as image:
            image.load()
            if (
                image.format != "PNG"
                or image.mode != mode
                or image.width != width
                or image.height != height
            ):
                raise ValueError(f"PNG metadata mismatch: {relative}")
    except OSError as exc:
        raise ValueError(f"PNG cannot be decoded: {relative}") from exc
    return relative, width, height, mode


def build(
    annotation_path: Path,
    conversion_path: Path,
    *,
    png_root: Path,
    manifest_root: Path,
) -> list[dict[str, object]]:
    png_root = png_root.resolve()
    manifest_root = manifest_root.resolve()
    try:
        png_prefix = PurePosixPath(png_root.relative_to(manifest_root).as_posix())
    except ValueError as exc:
        raise ValueError("PNG root must be inside the manifest directory.") from exc

    converted: dict[str, dict[str, Any]] = {}
    for row in _rows(conversion_path):
        source = _relative(row.get("source_relative_path"), "source_relative_path")
        if row.get("status") != "converted":
            continue
        if source in converted:
            raise ValueError(f"Duplicate successful source path: {source}")
        converted[source] = row

    output: list[dict[str, object]] = []
    image_ids: set[str] = set()
    sources: set[str] = set()
    png_paths: set[str] = set()
    for row in _rows(annotation_path):
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip() or image_id in image_ids:
            raise ValueError("annotation image_id values must be unique non-empty strings.")
        source = _relative(row.get("image_path"), "annotation.image_path")
        if source in sources:
            raise ValueError(f"Annotation has duplicate image_path: {source}")
        conversion = converted.get(source)
        if conversion is None:
            raise ValueError(f"No successful PNG conversion for annotation image_path: {source}")
        png, width, height, mode = verify_png(conversion, png_root)
        if png in png_paths:
            raise ValueError(f"Conversion manifest has duplicate PNG path: {png}")
        output.append(
            {
                "image_id": image_id,
                "source_path": source,
                "png_path": (png_prefix / png).as_posix(),
                "width": width,
                "height": height,
                "mode": mode,
                "sha256": conversion["output_sha256"],
                "conversion_status": "success",
                "transform": {"spatial": "none"},
            }
        )
        image_ids.add(image_id)
        sources.add(source)
        png_paths.add(png)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--png-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = build(
        args.annotation,
        args.conversion_manifest,
        png_root=args.png_root,
        manifest_root=args.output.parent,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {"records": len(rows), "modes": dict(Counter(str(row["mode"]) for row in rows))},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
