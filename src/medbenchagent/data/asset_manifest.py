"""Generic Asset manifest validation and deterministic identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from medbenchagent.data.sources import digest_file
from medbenchagent.data.models import AssetEntry


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


def iter_asset_manifest(path: str | Path, *, source_root: str | Path | None = None,
                        source_roots: dict[str, str | Path] | None = None):
    path = Path(path)
    if path.name != "asset_manifest.jsonl" or not path.is_file():
        raise ValueError("asset_manifest.jsonl is required")
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                entry = AssetEntry.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid asset manifest line {line_number}") from exc
            if entry.asset_id in seen:
                raise ValueError(f"Duplicate asset_id: {entry.asset_id}")
            seen.add(entry.asset_id)
            root_value = (source_roots or {}).get(entry.source_id, source_root)
            if root_value is not None:
                root = Path(root_value).resolve()
                resolved = (root / entry.source_path).resolve()
                if Path(entry.source_path).is_absolute() or not resolved.is_relative_to(root):
                    raise ValueError(f"Asset path escapes source root: {entry.source_path}")
                if not resolved.is_file() and not resolved.is_dir():
                    raise ValueError(f"Asset is missing or changed: {entry.source_path}")
                members = entry.metadata.get("members") if isinstance(entry.metadata, dict) else None
                actual_digest = digest_members(root, members) if members else digest_path(resolved)
                if actual_digest != entry.file_digest:
                    raise ValueError(f"Asset is missing or changed: {entry.source_path}")
            yield entry
    if not seen:
        raise ValueError("asset_manifest.jsonl must not be empty")


def write_asset_manifest(path: str | Path, entries: Iterable[AssetEntry]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json(exclude_none=True) + "\n")


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

