"""Stable run-directory initialization and Blueprint snapshot helpers."""

from __future__ import annotations

import json
import uuid
import shutil
from pathlib import Path

from medbenchagent.runtime.construction import record_event
from medbenchagent.settings import AppSettings, get_settings


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
    snapshot = active_settings.snapshot(graph_overrides)
    inputs = run_directory / ".runtime" / "inputs"
    inputs.mkdir(parents=True)
    for name in ("blueprint_template", "dataset_catalog"):
        source = getattr(active_settings.paths, name)
        if source.is_file():
            target = inputs / source.name
            shutil.copyfile(source, target)
            snapshot["settings"]["paths"][name] = str(target)
    materials = inputs / "materials"
    materials.mkdir()
    for source in active_settings.paths.materials_root.rglob("*"):
        if not source.is_file() or source.suffix.lower() != ".pdf":
            continue
        target = materials / source.relative_to(active_settings.paths.materials_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    snapshot["settings"]["paths"]["materials_root"] = str(materials)
    atomic_write_json(effective_config_path, snapshot)
    return {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "working_blueprint_path": str(run_directory / "blueprint.json"),
        "construction_events_path": str(events_path),
        "effective_config_path": str(effective_config_path),
    }
