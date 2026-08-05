"""Small file-writing helpers kept independent from the benchmark builder runtime."""

from __future__ import annotations

import json
from pathlib import Path


def atomic_write_json(path: str | Path, value: object) -> None:
    """Atomically write readable UTF-8 JSON without exposing a partial destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
