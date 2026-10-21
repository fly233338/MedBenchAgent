"""Persistent experiment conversations and their independent workflow runs."""

from __future__ import annotations

import json
import re
import shutil
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import tomlkit
from filelock import FileLock, Timeout

from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_settings


EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_path(name: str, root: Path = EXPERIMENTS_ROOT) -> Path:
    if not _NAME.fullmatch(name):
        raise ValueError("Experiment name must start with a letter and contain only lowercase letters, digits and hyphens")
    return root / name


def _merge_toml(target, source) -> None:
    for key, value in source.items():
        if key in target and isinstance(target[key], tomlkit.items.Table) and isinstance(value, tomlkit.items.Table):
            _merge_toml(target[key], value)
        else:
            target[key] = value


def create_experiment(name: str, *, root: Path = EXPERIMENTS_ROOT, template: Path | None = None) -> dict:
    folder = experiment_path(name, root)
    if folder.exists():
        raise ValueError(f"Experiment already exists: {name}")
    document = tomlkit.parse(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if template:
        _merge_toml(document, tomlkit.parse(template.read_text(encoding="utf-8")))
    folder.mkdir(parents=True)
    config = folder / "config.toml"
    config.write_text(tomlkit.dumps(document), encoding="utf-8")
    try:
        settings = load_settings(config)
        from medbenchagent.planning.bir import BIRStore
        BIRStore.load_template(settings.paths.bir_template)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    record = {"schema_version": "compilation-session", "name": name, "created_at": _now(),
              "messages": [], "runs": [], "pending_runs": []}
    atomic_write_json(folder / "session.json", record)
    return record


def load_experiment(name: str, *, root: Path = EXPERIMENTS_ROOT) -> dict:
    folder = experiment_path(name, root)
    path = folder / "session.json"
    if not path.is_file():
        raise ValueError(f"Unknown experiment: {name}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "compilation-session" or record.get("name") != name:
        raise ValueError(f"Invalid experiment session: {name}")
    return record




def remove_experiment(name: str, *, root: Path = EXPERIMENTS_ROOT) -> None:
    """Remove one idle experiment workspace without touching shared run or dataset roots."""
    folder = experiment_path(name, root)
    if folder.is_symlink() or folder.resolve().parent != root.resolve():
        raise ValueError("Experiment workspace must be a direct directory under the experiments root")
    if not folder.is_dir():
        raise ValueError(f"Unknown experiment: {name}")
    try:
        with ExitStack() as stack:
            for filename in ("submit.lock", "run.lock", "session.json.lock"):
                stack.enter_context(FileLock(str(folder / filename)).acquire(timeout=0))
            load_experiment(name, root=root)
    except Timeout as exc:
        raise ValueError(f"Experiment is in use: {name}") from exc
    shutil.rmtree(folder)


def save_experiment(record: dict, *, root: Path = EXPERIMENTS_ROOT) -> None:
    atomic_write_json(experiment_path(record["name"], root) / "session.json", record)


def add_message(record: dict, role: str, text: str, *, run_id: str | None = None,
                root: Path = EXPERIMENTS_ROOT, save: bool = True) -> dict:
    message = {"id": str(uuid4()), "role": role, "text": text, "at": _now(), "run_id": run_id}
    record["messages"].append(message)
    if save:
        save_experiment(record, root=root)
    return message


def queue_run(record: dict, requirement: str, *, root: Path = EXPERIMENTS_ROOT,
              save: bool = True, dataset_id: str | None = None) -> dict:
    from medbenchagent.planning.bir import BIRStore
    identifier = str(uuid4())
    folder = experiment_path(record["name"], root) / "compilations" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    config = folder / "config.toml"
    shutil.copy2(config_path(record["name"], root=root), config)
    settings = load_settings(config)
    store = BIRStore.initialize(folder / "BIR", settings.paths.bir_template, identifier)
    run = {"id": identifier, "compilation_id": identifier, "thread_id": identifier,
           "compilation_directory": str(folder), "config_path": str(config),
           "dataset_id": dataset_id, "requirement": requirement,
           "status": "queued", "run_id": None, "run_directory": None,
           "runs_root": None, "construction_events_path": None,
           "effective_config_path": None, "catalog_snapshot_path": None,
           "bir_path": str(store.root),
           "bir_manifest_digest": store.digest("manifest.json"),
           "summary": "", "created_at": _now()}
    record["runs"].append(run)
    record["pending_runs"].append(run["id"])
    if save:
        save_experiment(record, root=root)
    return run


def update_run(record: dict, session_run_id: str, *, root: Path = EXPERIMENTS_ROOT, **changes) -> dict:
    run = next(run for run in record["runs"] if run["id"] == session_run_id)
    run.update(changes)
    save_experiment(record, root=root)
    return run


def pop_pending(record: dict, *, root: Path = EXPERIMENTS_ROOT) -> dict | None:
    for identifier in list(record["pending_runs"]):
        run = next(run for run in record["runs"] if run["id"] == identifier)
        record["pending_runs"].remove(identifier)
        save_experiment(record, root=root)
        return run
    return None


def config_path(name: str, *, root: Path = EXPERIMENTS_ROOT) -> Path:
    return experiment_path(name, root) / "config.toml"


def config_values(name: str, *, root: Path = EXPERIMENTS_ROOT) -> dict:
    return tomlkit.parse(config_path(name, root=root).read_text(encoding="utf-8")).unwrap()


def update_config_value(name: str, keys: tuple[str, ...], value: object,
                        *, root: Path = EXPERIMENTS_ROOT) -> None:
    if not keys or keys[0] == "dataset_sources":
        raise ValueError("dataset_sources are managed by source discovery")
    path = config_path(name, root=root)
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    current = document
    for key in keys[:-1]:
        current = current[key]
    if value is None:
        current.pop(keys[-1], None)
    else:
        current[keys[-1]] = value
    temporary = path.with_suffix(".toml.tmp")
    try:
        temporary.write_text(tomlkit.dumps(document), encoding="utf-8")
        load_settings(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def upsert_dataset_source(name: str, entry: dict, *, root: Path = EXPERIMENTS_ROOT) -> None:
    """Write an exact, validated source once; conflicting IDs require a new decision."""
    from medbenchagent.data.contracts import SourcesArguments

    SourcesArguments.model_validate({"dataset_id": entry["dataset_id"], "sources": entry["sources"]})
    path = config_path(name, root=root)
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    existing = next((item for item in document["dataset_sources"]
                     if item["dataset_id"] == entry["dataset_id"]), None)
    if existing is not None:
        if existing.unwrap() == entry:
            return
        raise ValueError(f"Conflicting dataset_sources entry: {entry['dataset_id']}")
    document["dataset_sources"].append(entry)
    temporary = path.with_suffix(".toml.tmp")
    try:
        temporary.write_text(tomlkit.dumps(document), encoding="utf-8")
        load_settings(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
