"""Stable run-directory initialization and Blueprint snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path


RUNS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "runs"


def atomic_write_json(path: str | Path, value: object) -> None:
    """Atomically write readable UTF-8 JSON without exposing partial files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_run(runs_root: str | Path = RUNS_ROOT) -> dict[str, str]:
    """Create one UUID-named run directory before the first model call."""
    run_id = str(uuid.uuid4())
    run_directory = Path(runs_root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    events_path = run_directory / "construction_events.jsonl"
    events_path.touch(exist_ok=False)
    return {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "working_blueprint_path": str(run_directory / "blueprint.json"),
        "construction_events_path": str(events_path),
    }


def update_blueprint_dataset(blueprint_path: str | Path, dataset: dict[str, object]) -> None:
    """Replace the top-level public dataset snapshot in a working Blueprint."""
    path = Path(blueprint_path)
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    blueprint["dataset"] = dataset
    atomic_write_json(path, blueprint)


def blueprint_sha256(blueprint_path: str | Path) -> str:
    return hashlib.sha256(Path(blueprint_path).read_bytes()).hexdigest()
