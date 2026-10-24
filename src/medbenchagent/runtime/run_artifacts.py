"""Atomic JSON writing for UI records, dataset registration and retrieval caches."""
import json
from pathlib import Path


def runtime_directory(directory: str | Path) -> Path:
    """Resolve the sole frozen location, never infer an old on-disk layout."""
    value = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
    path = Path(value["runtime_directory"])
    if not path.is_absolute():
        raise ValueError("Run runtime_directory must be absolute")
    return path.resolve()


def separate_directories(data: Path, runtime: Path, sources=()) -> None:
    """Both output roots must be disjoint from each other and delegated sources."""
    data, runtime = data.resolve(), runtime.resolve()
    pairs = [(data, runtime), *((output, Path(source).resolve()) for source in sources for output in (data, runtime))]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in pairs):
        raise ValueError("Run data, runtime and original source directories must be separate")

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
