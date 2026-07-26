"""Validate canonical annotations and their prepared PNG manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from build_image_manifest import ALLOWED_MODES, HEX_SHA256, _relative
from PIL import Image

MANIFEST_FIELDS = {
    "image_id",
    "source_path",
    "png_path",
    "width",
    "height",
    "mode",
    "sha256",
    "conversion_status",
    "transform",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def validate(
    annotation_path: Path,
    manifest_path: Path,
    schema_path: Path,
) -> dict[str, object]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required_fields = schema.get("required")
    if not isinstance(required_fields, list) or not all(
        isinstance(field, str) for field in required_fields
    ):
        raise ValueError("Annotation schema has an invalid required-field contract.")
    required = set(required_fields)
    annotations = _rows(annotation_path)
    manifest = _rows(manifest_path)
    if len(annotations) != len(manifest):
        raise ValueError("Annotation and image-manifest counts differ.")

    root = manifest_path.parent.resolve()
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_pngs: set[str] = set()
    modes: Counter[str] = Counter()
    for index, (annotation, entry) in enumerate(zip(annotations, manifest, strict=True), start=1):
        if not required <= set(annotation):
            raise ValueError(f"Annotation row {index} is missing canonical fields.")
        image_id = annotation.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip() or image_id in seen_ids:
            raise ValueError(f"Annotation row {index} has an invalid image_id.")
        if not isinstance(annotation.get("annotation"), dict):
            raise ValueError(f"Annotation row {index} has a non-object annotation.")
        for field in ("patient_id", "case_id", "study_id"):
            value = annotation.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Annotation row {index} has an invalid {field}.")
        source_path = _relative(annotation.get("image_path"), f"annotation row {index} image_path")
        if source_path in seen_sources:
            raise ValueError(f"Annotation row {index} has a duplicate image_path.")

        if set(entry) != MANIFEST_FIELDS:
            raise ValueError(f"Manifest row {index} has an invalid shape.")
        if entry.get("image_id") != image_id or entry.get("source_path") != source_path:
            raise ValueError(f"Manifest row {index} does not match annotation order.")
        if entry.get("conversion_status") != "success" or entry.get("transform") != {
            "spatial": "none"
        }:
            raise ValueError(f"Manifest row {index} has an invalid conversion contract.")
        relative_png = _relative(entry.get("png_path"), f"manifest row {index} png_path")
        if relative_png in seen_pngs:
            raise ValueError(f"Manifest row {index} has a duplicate png_path.")
        png_path = root.joinpath(*PurePosixPath(relative_png).parts).resolve()
        try:
            png_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Manifest row {index} escapes the artifact directory.") from exc
        mode = entry.get("mode")
        digest = entry.get("sha256")
        width, height = entry.get("width"), entry.get("height")
        if (
            mode not in ALLOWED_MODES
            or not isinstance(digest, str)
            or not HEX_SHA256.fullmatch(digest)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or not png_path.is_file()
            or _sha256(png_path) != digest
        ):
            raise ValueError(f"Manifest row {index} has invalid PNG metadata.")
        try:
            with Image.open(png_path) as image:
                image.load()
                if image.format != "PNG" or image.mode != mode or image.size != (width, height):
                    raise ValueError(f"Manifest row {index} does not match its PNG.")
        except OSError as exc:
            raise ValueError(f"Manifest row {index} PNG cannot be decoded.") from exc
        seen_ids.add(image_id)
        seen_sources.add(source_path)
        seen_pngs.add(relative_png)
        modes[str(mode)] += 1

    return {
        "records": len(annotations),
        "modes": dict(sorted(modes.items())),
        "annotation_sha256": _sha256(annotation_path),
        "image_manifest_sha256": _sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
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
            validate(args.annotation, args.image_manifest, args.schema),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
