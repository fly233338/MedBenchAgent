"""Strict UTF-8 and deterministic digest primitives for preparation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from pydantic import BaseModel


def digest_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_payload(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def digest_tree(root: str | Path, *, exclude: set[str] | None = None) -> str:
    root = Path(root)
    ignored = exclude or set()
    return digest_payload({
        path.relative_to(root).as_posix(): digest_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not set(path.relative_to(root).parts).intersection(ignored)
    })


def relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts) or ":" in path.parts[0]:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def confined(root: str | Path, value: str, *, must_exist: bool = True) -> Path:
    root = Path(root).resolve()
    path = (root / relative_path(value)).resolve()
    if not path.is_relative_to(root) or (must_exist and not path.exists()):
        raise ValueError(f"Missing or external path: {value!r}")
    return path


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path, model: type[BaseModel] | None = None) -> Iterator[Any]:
    with Path(path).open(encoding="utf-8", errors="strict") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield model.model_validate_json(line) if model else json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"{Path(path).name}:{number}: {exc}") from exc


def write_jsonl(path: str | Path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
