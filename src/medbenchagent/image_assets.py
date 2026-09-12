"""Shared image mapping contract used by preparation, construction and evaluation."""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass


class ImageManifestError(ValueError):
    """An image mapping violates the standard dataset contract."""


@dataclass(frozen=True)
class ImageEntry:
    image_id: str
    png_path: Path
    relative_png_path: str
    width: int
    height: int


def load_image_manifest(path: Path) -> dict[str, ImageEntry]:
    root = path.parent.resolve()
    images: dict[str, ImageEntry] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ImageManifestError(f"Image manifest could not be read: {path}") from exc
    if not lines:
        raise ImageManifestError("Image manifest must not be empty.")
    expected = {"image_id", "png_path", "width", "height"}
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ImageManifestError(
                f"Image manifest line {line_number} is not valid JSON."
            ) from exc
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ImageManifestError(
                f"Image manifest line {line_number} must contain exactly {sorted(expected)}."
            )
        image_id, relative = raw["image_id"], raw["png_path"]
        width, height = raw["width"], raw["height"]
        if not isinstance(image_id, str) or not image_id or image_id in images:
            raise ImageManifestError(f"Image manifest line {line_number} has an invalid image_id.")
        if not isinstance(relative, str) or not relative:
            raise ImageManifestError(f"{image_id}: png_path must be a non-empty string.")
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ImageManifestError(f"{image_id}: png_path must be relative.")
        resolved = (root / relative_path).resolve()
        if not resolved.is_relative_to(root):
            raise ImageManifestError(f"{image_id}: png_path escapes the manifest directory.")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            raise ImageManifestError(f"{image_id}: invalid dimensions.")
        images[image_id] = ImageEntry(
            image_id=image_id,
            png_path=resolved,
            relative_png_path=relative_path.as_posix(),
            width=width,
            height=height,
        )
    if len({entry.png_path for entry in images.values()}) != len(images):
        raise ImageManifestError("Image manifest paths must be unique")
    return images
