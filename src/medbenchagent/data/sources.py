"""Content digests for source files and directory assets."""

import hashlib
from pathlib import Path


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
