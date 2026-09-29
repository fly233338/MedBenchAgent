"""Shared image mapping contract used by preparation, construction and evaluation."""

from __future__ import annotations
import json
from pathlib import Path, PurePosixPath
from typing import Any
from dataclasses import dataclass

from PIL import Image
from medbenchagent.data.sources import digest_file


class ImageManifestError(ValueError):
    """An image mapping violates the standard dataset contract."""


@dataclass(frozen=True)
class ImageEntry:
    image_id: str
    png_path: Path
    relative_png_path: str
    width: int
    height: int
    source_id: str | None = None
    source_path: str | None = None
    file_digest: str | None = None


def resolve_source_image(source_id: str, source_path: str, source_roots: dict[str, str], expected_digest: str | None = None) -> Path:
    """Resolve an external read-only image and verify its optional digest."""
    if source_id not in source_roots:
        raise ImageManifestError(f"Unknown image source: {source_id}")
    relative = Path(source_path)
    root = Path(source_roots[source_id]).resolve()
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ImageManifestError(f"Image source path is missing or escapes its source: {source_id}/{source_path}")
    if expected_digest and digest_file(path) != expected_digest:
        raise ImageManifestError(f"Image source digest changed: {source_id}/{source_path}")
    return path


def iter_image_manifest(path: Path, *, source_roots: dict[str, str] | None = None):
    """Validate the manifest incrementally with a disk-backed uniqueness index."""
    import sqlite3
    import tempfile
    root = path.parent.resolve()
    if not path.is_file():
        raise ImageManifestError(f"Image manifest could not be read: {path}")
    with tempfile.TemporaryDirectory(prefix="medbench-manifest-") as temporary:
        db = sqlite3.connect(str(Path(temporary) / "keys.sqlite"))
        try:
            db.execute("CREATE TABLE seen (id TEXT PRIMARY KEY, path TEXT UNIQUE)")
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ImageManifestError(f"Image manifest line {line_number} is not valid JSON") from exc
                    external = isinstance(raw, dict) and "source_id" in raw
                    expected_keys = {"image_id", "source_id", "source_path", "width", "height", "file_digest"} if external else {"image_id", "png_path", "width", "height"}
                    if not isinstance(raw, dict) or set(raw) != expected_keys:
                        raise ImageManifestError("Image manifest has an invalid schema")
                    image_id = raw["image_id"]
                    relative = raw.get("source_path", raw.get("png_path"))
                    width, height = raw["width"], raw["height"]
                    if not isinstance(image_id, str) or not image_id.strip():
                        raise ImageManifestError("Invalid image_id")
                    if not isinstance(relative, str) or not relative:
                        raise ImageManifestError("png_path must be a non-empty string")
                    relative_path = Path(relative)
                    if external:
                        if source_roots is None:
                            raise ImageManifestError("External image manifest requires source roots")
                        resolved = resolve_source_image(raw["source_id"], relative, source_roots, raw["file_digest"])
                    else:
                        resolved = (root / relative_path).resolve()
                        if relative_path.is_absolute() or not resolved.is_relative_to(root):
                            raise ImageManifestError("png_path escapes the manifest directory")
                    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
                        raise ImageManifestError("Invalid image dimensions")
                    try:
                        db.execute("INSERT INTO seen VALUES (?,?)", [image_id, str(resolved)])
                    except sqlite3.IntegrityError as exc:
                        raise ImageManifestError("Image manifest IDs and paths must be unique") from exc
                    count += 1
                    yield ImageEntry(image_id, resolved, relative_path.as_posix(), width, height,
                                     raw.get("source_id"), relative if external else None,
                                     raw.get("file_digest"))
            if not count:
                raise ImageManifestError("Image manifest must not be empty")
        finally:
            db.close()


def load_image_manifest(path: Path, *, source_roots: dict[str, str] | None = None) -> dict[str, ImageEntry]:
    """Materialize the manifest for consumers requiring random image access."""
    return {entry.image_id: entry for entry in iter_image_manifest(path, source_roots=source_roots)}


def validate_image_asset(
    entry: dict[str, Any], root: Path, *,
    check_uniform: bool = True, supported_formats: frozenset[str] | None = None,
    source_roots: dict[str, str] | None = None,
) -> list[str]:
    """Check a referenced image; consumers choose pixel and encoding policy."""
    if entry.get("source_id") is not None:
        try:
            image_path = resolve_source_image(entry["source_id"], entry["source_path"], source_roots or {}, entry.get("file_digest"))
        except ImageManifestError:
            return ["image_manifest_path"]
        png = entry.get("source_path")
    else:
        png = entry.get("png_path")
    if not isinstance(png, str):
        return ["image_manifest_path"]
    path = PurePosixPath(png)
    if path.is_absolute() or ".." in path.parts:
        return ["image_manifest_path"]
    if entry.get("source_id") is None:
        image_path = root.joinpath(*path.parts).resolve()
        if not image_path.is_relative_to(root.resolve()):
            return ["image_manifest_path"]
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return ["image_missing"]
    try:
        with Image.open(image_path) as image:
            image.load()
            if supported_formats is not None and image.format not in supported_formats:
                return ["image_format"]
            if (
                image.width <= 0
                or image.height <= 0
                or image.width != entry.get("width")
                or image.height != entry.get("height")
            ):
                return ["image_dimensions"]
            if check_uniform:
                extrema = image.getextrema()
                channels = extrema if isinstance(extrema[0], tuple) else (extrema,)
                if all(minimum == maximum for minimum, maximum in channels):
                    return ["image_pixels"]
    except (OSError, ValueError):
        return ["image_decode"]
    return []
