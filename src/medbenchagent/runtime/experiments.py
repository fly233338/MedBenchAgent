"""Names for persistent LangGraph threads; checkpoints remain the execution authority."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from medbenchagent.runtime.run_artifacts import atomic_write_json


def registry_path(runs_root):
    return Path(runs_root).parent / "experiments.json"


def _read(runs_root):
    path = registry_path(runs_root)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"experiments": {}}


def _now():
    return datetime.now(timezone.utc).isoformat()


def list_experiments(runs_root):
    return sorted(_read(runs_root)["experiments"].values(), key=lambda e: e["updated_at"], reverse=True)


def get_experiment(runs_root, name):
    entry = _read(runs_root)["experiments"].get(name)
    if entry is None:
        raise ValueError(f"Unknown experiment: {name}")
    return entry


def create_experiment(runs_root, name, *, dataset_id=None, dataset_ids=None):
    name = name.strip()
    if not name or name == "new" or any(c.isspace() for c in name):
        raise ValueError("Experiment name must be nonempty without whitespace; 'new' is reserved")
    if dataset_id and dataset_ids:
        raise ValueError("Preparation experiments do not select benchmark datasets")
    if dataset_id:
        from medbenchagent.data.dataset_catalog import validate_dataset_id
        validate_dataset_id(dataset_id)
    document = _read(runs_root)
    if name in document["experiments"]:
        raise ValueError(f"Experiment already exists: {name}")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    now = _now()
    entry = {"name": name, "thread_id": str(uuid4()), "mode": "prepare" if dataset_id else "auto",
             "dataset_id": dataset_id, "requested_dataset_ids": dataset_ids, "datasets": [],
             "run_id": None, "run_directory": None, "status": "created", "git_commit": commit,
             "created_at": now, "updated_at": now}
    document["experiments"][name] = entry
    atomic_write_json(registry_path(runs_root), document)
    return entry


def update_experiment(runs_root, name, **updates):
    document = _read(runs_root)
    if name not in document["experiments"]:
        raise ValueError(f"Unknown experiment: {name}")
    entry = document["experiments"][name]
    if set(updates) - {"mode", "datasets", "run_id", "run_directory", "status"}:
        raise ValueError("Experiment identity cannot be changed")
    entry.update(updates, updated_at=_now())
    atomic_write_json(registry_path(runs_root), document)
    return entry


def snapshot_updates(snapshot, *, interrupted=False):
    values = snapshot.values or {}
    context = values.get("context", {})
    waiting = any(task.interrupts for task in getattr(snapshot, "tasks", ()))
    if waiting:
        status = "waiting_input"
    elif interrupted:
        status = "interrupted"
    elif values.get("status") in {"complete", "blocked", "failed"}:
        status = values["status"]
    elif snapshot.next:
        status = "resumable"
    else:
        status = "created" if not values else "failed"
    intent = context.get("run_intent", {}).get("intent")
    updates = {"status": status,
               "datasets": list(dict.fromkeys(context.get("selected_dataset_ids", []) +
                                               list(context.get("prepared_entries", {}))))}
    if intent:
        updates["mode"] = "prepare" if intent == "prepare_dataset" else "build"
    for key in ("run_id", "run_directory"):
        if value := values.get(key) or context.get(key):
            updates[key] = value
    return updates
