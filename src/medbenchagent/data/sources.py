"""Content digests for source files and directory assets."""

import hashlib
from pathlib import Path


def inspect_source(root, *, relative=".", offset=0, limit=20):
    """Inspect one bounded directory page; no persisted inventory or recursive scan."""
    from .asset_adapters import detect_format
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if Path(relative).is_absolute() or not target.is_relative_to(root) or not target.is_dir():
        raise ValueError("Source directory must resolve beneath its declared root")
    if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
        raise ValueError("Invalid source page")
    entries = sorted(target.iterdir(), key=lambda path: path.name)
    result = []
    for path in entries[offset:offset + limit]:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Source entry escapes its declared root")
        row = {"path": path.relative_to(root).as_posix(), "directory": path.is_dir()}
        if path.is_file():
            row.update(size=path.stat().st_size, digest=digest_file(path),
                       format=detect_format(path) or path.suffix.lower().lstrip(".") or "unknown")
        result.append(row)
    return {"root": str(root), "directory": target.relative_to(root).as_posix(), "coverage": "one directory page",
            "entries": result, "next_offset": offset + limit if offset + limit < len(entries) else None}


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_path(path: str | Path) -> str:
    """Hash a file or directory with sorted relative names and file bytes."""
    path = Path(path)
    if path.is_file():
        return digest_file(path)
    if not path.is_dir():
        raise ValueError(f"Asset path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
