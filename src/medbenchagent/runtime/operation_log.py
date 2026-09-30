"""Compact durable operation receipts for a run."""

from __future__ import annotations

import json
from pathlib import Path

from filelock import FileLock


def path_for(run_directory: str | Path) -> Path:
    return Path(run_directory) / "control" / "operations.jsonl"


def read(path: str | Path, operation_id: str) -> dict | None:
    destination = Path(path)
    if not destination.is_file():
        return None
    found = None
    with destination.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(item, dict) and item.get("id") == operation_id:
                found = item
    return found


def append(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(destination) + ".lock")
    with lock:
        with destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
