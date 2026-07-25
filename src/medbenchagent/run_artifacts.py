"""Stable run-directory initialization and Blueprint snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from medbenchagent.construction import record_event
from medbenchagent.settings import AppSettings, get_settings


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


def initialize_run(
    runs_root: str | Path | None = None,
    *,
    settings: AppSettings | None = None,
    graph_overrides: dict[str, object] | None = None,
) -> dict[str, str]:
    """Create one UUID-named run directory before the first model call."""
    active_settings = settings or get_settings()
    run_id = str(uuid.uuid4())
    run_directory = Path(runs_root) if runs_root is not None else active_settings.paths.runs_root
    run_directory = run_directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    events_path = run_directory / "construction_events.jsonl"
    events_path.touch(exist_ok=False)
    record_event(
        events_path,
        "run_started",
        run_id=run_id,
        operation="initialize",
        component="orchestrator",
        status="started",
    )
    effective_config_path = run_directory / "effective_config.json"
    atomic_write_json(effective_config_path, active_settings.snapshot(graph_overrides))
    return {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "working_blueprint_path": str(run_directory / "blueprint.json"),
        "construction_events_path": str(events_path),
        "effective_config_path": str(effective_config_path),
    }


def update_blueprint_dataset(blueprint_path: str | Path, dataset_id: str) -> None:
    """Persist the selected dataset identifier as the complete Blueprint dataset section."""
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")
    path = Path(blueprint_path)
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    blueprint["dataset"] = {"dataset_id": dataset_id.strip()}
    atomic_write_json(path, blueprint)


def blueprint_sha256(blueprint_path: str | Path) -> str:
    return hashlib.sha256(Path(blueprint_path).read_bytes()).hexdigest()
