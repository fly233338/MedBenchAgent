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
    source = active_settings.paths.blueprint_template
    if source.is_file():
        target = inputs / source.name
        shutil.copyfile(source, target)
        snapshot["settings"]["paths"]["blueprint_template"] = str(target)
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
    from medbenchagent.data.dataset_catalog import load_dataset_catalog, resolve_path
    from medbenchagent.data.sources import digest_file

    frozen_catalog = []
    card_digests = {}
    cards = inputs / "candidate_cards"
    for entry in load_dataset_catalog(active_settings.paths.dataset_catalog):
        frozen = dict(entry)
        original = resolve_path(entry["dataset_card_path"])
        if original.is_file():
            cards.mkdir(exist_ok=True)
            target = cards / f"{entry['dataset_id']}.json"
            shutil.copyfile(original, target)
            frozen["dataset_card_path"] = str(target)
            card_digests[entry["dataset_id"]] = digest_file(target)
        frozen_catalog.append(frozen)
    catalog_snapshot_path = inputs / "catalog_snapshot.json"
    atomic_write_json(catalog_snapshot_path, {"entries": frozen_catalog, "card_digests": card_digests})
    return {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "construction_events_path": str(events_path),
        "effective_config_path": str(effective_config_path),
        "task_catalog_snapshot_path": str(catalog_snapshot_path),
        "task_catalog_snapshot_digest": digest_file(catalog_snapshot_path),
    }


def verify_catalog_snapshot(path: str | Path, expected_digest: str) -> None:
    """Reject changes to a run's original candidate universe or copied Cards."""
    from medbenchagent.data.sources import digest_file

    frozen = Path(path)
    if not frozen.is_file() or digest_file(frozen) != expected_digest:
        raise ValueError("Frozen task catalog changed or is unavailable")
    document = json.loads(frozen.read_text(encoding="utf-8"))
    entries = {entry["dataset_id"]: entry for entry in document["entries"]}
    for dataset_id, digest in document["card_digests"].items():
        card = Path(entries[dataset_id]["dataset_card_path"])
        if not card.is_file() or digest_file(card) != digest:
            raise ValueError(f"Frozen candidate Card changed: {dataset_id}")
