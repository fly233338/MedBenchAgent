"""Atomic JSON writing for UI records, dataset registration and retrieval caches."""
import json
from pathlib import Path

def atomic_write_json(path: str | Path, value: object) -> None:
    """Atomically write readable UTF-8 JSON without exposing partial files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
