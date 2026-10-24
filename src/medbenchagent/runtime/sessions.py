"""User-facing Runs; UI records never duplicate Compilation execution state."""

import json
import re
from pathlib import Path

import tomlkit
from filelock import FileLock

from medbenchagent.settings import DEFAULT_CONFIG_PATH, load_paths, local_config_path, _merge
from medbenchagent.runtime.run_artifacts import (
    atomic_write_json,
    runtime_directory,
    separate_directories,
)


def run_path(name, root=None):
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise ValueError("Use a-z, 0-9 and hyphens. Start with a letter.")
    return (Path(root) if root is not None else load_paths()["runs_root"]).resolve() / name


def session_directory(directory):
    directory = Path(directory).resolve()
    if (directory / "run.json").is_file():
        return runtime_directory(directory)
    return load_paths(directory / "config.toml")["runtime_root"] / directory.name


def create_session(name, *, root=None, runtime_root=None, template=None):
    selected = template or local_config_path()
    paths = load_paths(selected)
    directory = run_path(name, root or paths["runs_root"])
    if directory.exists():
        raise ValueError("Run already exists")
    document = tomlkit.parse(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if selected is not None:
        _merge(document, tomlkit.parse(Path(selected).read_text(encoding="utf-8")))
    run_root = directory.parent
    runtime_root = Path(runtime_root or paths["runtime_root"]).resolve()
    runtime = runtime_root / name
    separate_directories(directory, runtime)
    if runtime.exists():
        raise ValueError("Runtime location already exists; choose another Run name")
    document["paths"] = {"runs_root": str(run_root), "runtime_root": str(runtime_root)}
    record = {
        "id": name,
        "name": name,
        "directory": str(directory),
        "runtime_directory": str(runtime),
        "messages": [],
        "draft": {"language": str(document["benchmark"]["default_language"])},
    }
    directory.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (runtime / "locks").mkdir()
    (directory / "config.toml").write_text(tomlkit.dumps(document), encoding="utf-8")
    atomic_write_json(runtime / "session.json", record)
    return record


def load_session(directory):
    directory = Path(directory).resolve()
    runtime = session_directory(directory)
    record = json.loads((runtime / "session.json").read_text(encoding="utf-8"))
    if (
        Path(record["directory"]).resolve() != directory
        or Path(record["runtime_directory"]).resolve() != runtime
    ):
        raise ValueError("Run identity does not match its stored locations")
    return record


def list_runs(root=None):
    root = Path(root) if root is not None else load_paths()["runs_root"]
    records = []
    for path in sorted(root.glob("*/config.toml")):
        try:
            record = load_session(path.parent)
            records.append({**record, "frozen": (path.parent / "run.json").is_file()})
        except (ValueError, OSError, KeyError):
            continue
    return records


def save_draft(directory, draft):
    directory = Path(directory)
    if (directory / "run.json").exists():
        raise ValueError("Run frozen. Clone to edit.")
    runtime = session_directory(directory)
    with FileLock(runtime / "locks/session.lock"):
        record = load_session(directory)
        record["draft"] = draft.model_dump(mode="json")
        atomic_write_json(runtime / "session.json", record)


def add_message(directory, message):
    runtime = session_directory(directory)
    with FileLock(runtime / "locks/session.lock"):
        record = load_session(directory)
        if not any(row.get("id") == message.get("id") for row in record["messages"]):
            record["messages"].append(message)
            atomic_write_json(runtime / "session.json", record)


def copy_session(directory, name, *, root=None, runtime_root=None):
    from medbenchagent.runtime.intake import InputDraft

    source = load_session(directory)
    record = create_session(
        name,
        root=root,
        runtime_root=runtime_root,
        template=Path(directory) / "config.toml",
    )
    save_draft(record["directory"], InputDraft.model_validate(source["draft"]))
    return load_session(record["directory"])


def freeze_session(directory, draft=None, **options):
    from medbenchagent.compilation.run import create_run
    from medbenchagent.runtime.intake import InputDraft, prepare_input

    directory = Path(directory).resolve()
    record = load_session(directory)
    runtime = Path(record["runtime_directory"])
    with FileLock(runtime / "locks/creation.lock", timeout=0):
        if (directory / "run.json").exists():
            raise ValueError("Run already frozen; resume it or copy its configuration")
        draft = draft or InputDraft.model_validate(record["draft"])
        save_draft(directory, draft)
        inputs = prepare_input(draft, directory) if not options else options
        result = create_run(
            directory,
            compilation_id=record["id"],
            goal=draft.goal,
            config_path=directory / "config.toml",
            runtime_directory=runtime,
            **inputs,
        )
        add_message(directory, {"id": "goal", "role": "user", "text": draft.goal})
    return result
