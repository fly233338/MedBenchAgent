"""Source-relative path containment."""

from pathlib import Path


def source_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.exists():
        raise ValueError(f"Source reference must resolve beneath source root: {relative}")
    return path


source_image_path = source_path
