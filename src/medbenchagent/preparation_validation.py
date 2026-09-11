"""Run inside the data container after conversion, independently of the generated script."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


def validate(source_root: Path, output_root: Path, converter: Path) -> None:
    spec = importlib.util.spec_from_file_location("conversion", converter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    annotations = [json.loads(line) for line in (output_root / "annotation.jsonl").read_text().splitlines()]
    images = {row["image_id"]: row for row in
              (json.loads(line) for line in (output_root / "image_manifest.jsonl").read_text().splitlines())}
    for row in annotations:
        source = (source_root / row["image"]["image_path"]).resolve()
        if not source.is_relative_to(source_root) or not source.is_file():
            raise ValueError("annotation image_path must reference the original source image")
        entry = images[row["image"]["image_id"]]
        target = (output_root / entry["png_path"]).resolve()
        if not target.is_relative_to(output_root):
            raise ValueError("Image output escapes preparation directory")
        # The existing format handlers reject volumes, multiframe and whole-slide images.
        detected = module.detect_format(source)
        if detected and detected.startswith("raster:"):
            with Image.open(source) as original:
                if getattr(original, "n_frames", 1) != 1 or (original.format == "TIFF" and original.tag_v2.get(330)):
                    raise ValueError("Multiframe or pyramidal images are unsupported")
                original_size = original.size
        else:
            original, _, _ = module.convert_image(source)
            original_size = original.size
        with Image.open(target) as image:
            if image.size != original_size:
                raise ValueError("Image geometry changed during conversion")
        if detected in {"raster:png", "raster:jpeg"} and source.read_bytes() != target.read_bytes():
            raise ValueError("PNG/JPEG source bytes must be preserved")


if __name__ == "__main__":
    validate(Path("/source"), Path("/work"), Path("/conversion.py"))
