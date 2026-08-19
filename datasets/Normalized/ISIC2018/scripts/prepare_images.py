"""Copy annotation-referenced ISIC2018 JPEGs without re-encoding.

Run this script manually on the server. It processes only image paths present in
annotation.jsonl and writes original JPEGs beneath the artifact's images/
directory. It never discovers or copies unrelated source files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


class ImagePreparationError(ValueError):
    """Raised when an annotation-referenced image cannot be prepared safely."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise ImagePreparationError(f"{label} does not exist: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ImagePreparationError(f"{label} does not exist: {path}")
    return path


def resolve_source_path(source_root: Path, relative_path: str, image_id: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ImagePreparationError(f"{image_id}: image_path must be relative")
    resolved = (source_root / candidate).resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ImagePreparationError(f"{image_id}: image_path escapes source root") from exc
    return require_file(resolved, f"Source image for {image_id}")


def load_annotations(annotation_path: Path) -> list[dict[str, Any]]:
    require_file(annotation_path, "annotation.jsonl")
    records: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    image_paths: set[str] = set()
    with annotation_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ImagePreparationError(f"annotation line {line_number}: blank line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ImagePreparationError(
                    f"annotation line {line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict) or set(record) != {"patient", "case", "study", "image"}:
                raise ImagePreparationError(f"annotation line {line_number}: invalid record")
            image = record.get("image")
            if not isinstance(image, dict):
                raise ImagePreparationError(f"annotation line {line_number}: invalid image object")
            image_id = image.get("image_id")
            image_path = image.get("image_path")
            if (
                not isinstance(image_id, str)
                or not image_id
                or image_id in image_ids
                or any(separator in image_id for separator in ("/", "\\"))
            ):
                raise ImagePreparationError(f"annotation line {line_number}: invalid/duplicate image_id")
            if not isinstance(image_path, str) or not image_path or image_path in image_paths:
                raise ImagePreparationError(f"annotation line {line_number}: invalid/duplicate image_path")
            annotation = image.get("annotation")
            if not isinstance(annotation, dict):
                raise ImagePreparationError(f"annotation line {line_number}: invalid image.annotation")
            width, height = annotation.get("width"), annotation.get("height")
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or width <= 0
                or not isinstance(height, int)
                or isinstance(height, bool)
                or height <= 0
            ):
                raise ImagePreparationError(
                    f"annotation line {line_number}: image.annotation width/height are required"
                )
            image_ids.add(image_id)
            image_paths.add(image_path)
            records.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                }
            )
    if not records:
        raise ImagePreparationError("annotation.jsonl contains no image records")
    return records


def copy_original_jpeg(
    source_path: Path, output_path: Path, expected_size: tuple[int, int]
) -> None:
    """Copy one RGB JPEG byte-for-byte without spatial or pixel transforms."""

    try:
        with Image.open(source_path) as source:
            if source.format != "JPEG":
                raise ImagePreparationError(
                    f"{source_path.name}: expected JPEG, detected {source.format}"
                )
            if getattr(source, "n_frames", 1) != 1:
                raise ImagePreparationError(f"{source_path.name}: multi-frame JPEG is unsupported")
            source.load()
            if source.size != expected_size:
                raise ImagePreparationError(
                    f"{source_path.name}: source size {source.size} != annotation {expected_size}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output_path)
    except OSError as exc:
        raise ImagePreparationError(f"Could not decode or write {source_path}") from exc


def validate_jpeg(path: Path, expected_size: tuple[int, int], image_id: str) -> None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.size != expected_size:
                raise ImagePreparationError(
                    f"{image_id}: JPEG format/size mismatch for {path.name}"
                )
            minimum, maximum = image.convert("L").getextrema()
            if minimum == maximum:
                raise ImagePreparationError(f"{image_id}: output JPEG has constant pixels")
    except OSError as exc:
        raise ImagePreparationError(f"{image_id}: output PNG could not be decoded") from exc


def manifest_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")


def publish_file(staged: Path, final: Path) -> None:
    if final.exists():
        if not final.is_file() or final.read_bytes() != staged.read_bytes():
            raise ImagePreparationError(f"Refusing to overwrite a different output: {final}")
        staged.unlink()
        return
    os.replace(staged, final)


def write_manifest_if_identical_or_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ImagePreparationError(f"Refusing to overwrite a different manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--image-output-dir",
        required=True,
        type=Path,
        help="Image directory beneath the annotation artifact, normally <artifact>/images.",
    )
    parser.add_argument(
        "--manifest-path",
        required=True,
        type=Path,
        help="image_manifest.jsonl beside annotation.jsonl; png_path remains a legacy field name.",
    )
    parser.add_argument(
        "--encoding",
        required=True,
        choices=("original_jpeg",),
        help="Copy the source JPEG bytes without re-encoding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation_path = args.annotation.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    image_output_dir = args.image_output_dir.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()
    require_directory(source_root, "source root")
    require_file(annotation_path, "annotation.jsonl")
    if args.encoding != "original_jpeg":
        raise ImagePreparationError(f"Unsupported encoding: {args.encoding}")
    artifact_root = manifest_path.parent
    if annotation_path.parent != artifact_root:
        raise ImagePreparationError(
            "annotation must be beside image_manifest.jsonl in the artifact directory"
        )
    if not image_output_dir.is_relative_to(artifact_root) or image_output_dir == artifact_root:
        raise ImagePreparationError(
            "image-output-dir must be a child directory of the manifest directory"
        )

    records = load_annotations(annotation_path)
    image_output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{image_output_dir.name}.staging-", dir=image_output_dir.parent
        )
    )
    staged_rows: list[dict[str, Any]] = []
    try:
        staged_files: list[tuple[Path, Path]] = []
        for record in records:
            image_id = record["image_id"]
            source_path = resolve_source_path(source_root, record["image_path"], image_id)
            staged_path = staging_dir / f"{image_id}.jpg"
            final_path = image_output_dir / f"{image_id}.jpg"
            expected_size = (record["width"], record["height"])
            copy_original_jpeg(source_path, staged_path, expected_size)
            validate_jpeg(staged_path, expected_size, image_id)
            staged_files.append((staged_path, final_path))
            staged_rows.append(
                {
                    "image_id": image_id,
                    "png_path": final_path.relative_to(artifact_root).as_posix(),
                    "width": expected_size[0],
                    "height": expected_size[1],
                    "sha256": sha256_file(staged_path),
                }
            )

        payload = manifest_bytes(staged_rows)
        if manifest_path.exists() and manifest_path.read_bytes() != payload:
            raise ImagePreparationError(f"Refusing to overwrite a different manifest: {manifest_path}")
        for staged_path, final_path in staged_files:
            if final_path.exists() and final_path.read_bytes() != staged_path.read_bytes():
                raise ImagePreparationError(f"Refusing to overwrite a different output: {final_path}")
        for staged_path, final_path in staged_files:
            publish_file(staged_path, final_path)
        write_manifest_if_identical_or_new(manifest_path, payload)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    manifest_rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    for row in manifest_rows:
        final_path = manifest_path.parent / row["png_path"]
        validate_jpeg(final_path, (row["width"], row["height"]), row["image_id"])
        if sha256_file(final_path) != row["sha256"]:
            raise ImagePreparationError(f"{row['image_id']}: final JPEG hash mismatch")
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "image_output_dir": str(image_output_dir),
                "images": len(manifest_rows),
                "encoding": args.encoding,
                "rerun_policy": "existing identical JPEGs and manifest are accepted; different outputs are rejected",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImagePreparationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
