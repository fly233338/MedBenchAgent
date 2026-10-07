"""Read-only source identities and explicit annotation provenance."""

import hashlib
import json
import os
import sqlite3
import filecmp
import shutil
from pathlib import Path
from typing import Iterable
from medbenchagent.data.models import AssetEntry

from medbenchagent.data.contracts import Source
from medbenchagent.runtime.run_artifacts import atomic_write_json


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compare_outputs(first, second):
    def files(folder):
        return sorted(
            p.relative_to(folder)
            for p in folder.rglob("*")
            if p.is_file()
            and not {"scratch", "cache"}.intersection(p.relative_to(folder).parts)
            and p.name not in {"upstream_provenance.jsonl", "preparation.json", "fallback.py"}
        )

    names = files(first)
    if names != files(second) or any(
        not filecmp.cmp(first / p, second / p, shallow=False) for p in names
    ):
        raise ValueError("Preparation is not deterministic")


def remove_work_directory(path, parent):
    target, root = Path(path).resolve(), Path(parent).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("Temporary directory must be beneath its operation directory")
    # A removed repeat directory must not retain a completion receipt claiming its outputs exist.
    # Delete the receipt first: interruption during cleanup then permits a clean repeat attempt.
    for receipt in target.parent.glob(target.name + "-*.complete.json"):
        receipt.unlink()
    shutil.rmtree(target)


def namespace_id(source, entity, value):
    if value is None:
        return None
    return json.dumps(
        [source.id, source.version, entity, str(value)], ensure_ascii=False, separators=(",", ":")
    )


def inventory_source(source: Source, destination):
    root = Path(source.root).resolve()
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    listing = destination / "files.jsonl"
    if all(
        (destination / name).is_file() for name in ("files.jsonl", "source.json", "used.sqlite")
    ):
        saved = Source.model_validate_json(
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
    with sqlite3.connect(destination / "used.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS used (path TEXT PRIMARY KEY, digest TEXT NOT NULL)")
    return destination


def track_files(manifest, paths):
    manifest = Path(manifest)
    source = Source.model_validate_json((manifest / "source.json").read_text(encoding="utf-8"))
    with sqlite3.connect(manifest / "used.sqlite") as db:
        for relative in paths:
            key = Path(relative).as_posix()
            digest = digest_file(source_path(source.root, relative))
            saved = db.execute("SELECT digest FROM used WHERE path=?", [key]).fetchone()
            if saved and saved[0] != digest:
                raise ValueError(f"Source changed: {key}; rebuild dependent artifacts")
            db.execute("INSERT OR IGNORE INTO used VALUES (?,?)", [key, digest])


def verify_source(manifest):
    manifest = Path(manifest)
    source = Source.model_validate_json((manifest / "source.json").read_text(encoding="utf-8"))
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
    with sqlite3.connect(manifest / "used.sqlite") as db:
        for relative, digest in db.execute("SELECT path,digest FROM used ORDER BY path"):
            if digest_file(source_path(root, relative)) != digest:
                raise ValueError(f"Source content changed: {relative}")


def source_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.exists():
        raise ValueError(f"Source reference must resolve beneath source root: {relative}")
    return path




def stable_id(*parts: object, prefix: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


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


def digest_members(root: str | Path, members: Iterable[str]) -> str:
    """Digest exactly the declared members of a directory-backed Asset."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    normalized = sorted({Path(member).as_posix() for member in members})
    if not normalized:
        raise ValueError("Asset member list must not be empty")
    for member in normalized:
        path = (root / member).resolve()
        if Path(member).is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Asset member is outside or missing: {member}")
        digest.update(member.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def group_dicom_series(root: str | Path, *, source_id: str = "source") -> list[AssetEntry]:
    """Group readable DICOM files by StudyInstanceUID/SeriesInstanceUID."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required for DICOM grouping") from exc
    root = Path(root).resolve()
    groups: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception:
            continue
        study = str(getattr(ds, "StudyInstanceUID", ""))
        series = str(getattr(ds, "SeriesInstanceUID", ""))
        if study and series:
            groups.setdefault((study, series), []).append(path)
    entries = []
    for (study, series), paths in sorted(groups.items()):
        relative = Path(paths[0]).parent.relative_to(root).as_posix() or "."
        entries.append(AssetEntry(
            asset_id=stable_id(source_id, study, series, prefix="asset"),
            kind="volume" if len(paths) > 1 else "image", format="dicom_series" if len(paths) > 1 else "dicom",
            source_path=relative, file_digest=digest_members(root, [p.relative_to(root).as_posix() for p in paths]),
            metadata={"study_uid": study, "series_uid": series,
                      "members": [path.relative_to(root).as_posix() for path in paths]} if len(paths) > 1 else {},
        ))
    return entries
