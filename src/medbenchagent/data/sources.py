"""Read-only source identities and explicit annotation provenance."""

import hashlib
import json
import os
from pathlib import Path

from medbenchagent.data.dataset import DatasetSource
from medbenchagent.runtime.run_artifacts import atomic_write_json


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()








def inventory_source(source: DatasetSource, destination):
    root = Path(source.root).resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    listing = destination / "files.jsonl"
    if all(
        (destination / name).is_file() for name in ("files.jsonl", "source.json")
    ):
        saved = DatasetSource.model_validate_json(
            (destination / "source.json").read_text(encoding="utf-8")
        )
        if saved != source:
            raise ValueError("Source identity changed; use a new input version")
        verify_source(destination)
        return destination
    with listing.open("w", encoding="utf-8") as output:
        for directory, directories, files in os.walk(root):
            directories.sort()
            for name in sorted(files):
                path = Path(directory) / name
                resolved = source_path(root, path.relative_to(root))
                stat = resolved.stat()
                output.write(
                    json.dumps(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    )
                    + "\n"
                )
    atomic_write_json(destination / "source.json", source.model_dump())
    return destination




def verify_source(manifest):
    manifest = Path(manifest)
    source = DatasetSource.model_validate_json((manifest / "source.json").read_text(encoding="utf-8"))
    root = Path(source.root)
    count = 0
    with (manifest / "files.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            path = source_path(root, item["path"])
            stat = path.stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"Source changed: {item['path']}; start a new input version")
            count += 1
    if sum(p.is_file() for p in root.rglob("*")) != count:
        raise ValueError("Source file population changed")


def source_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.exists():
        raise ValueError(f"Source reference must resolve beneath source root: {relative}")
    return path






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
