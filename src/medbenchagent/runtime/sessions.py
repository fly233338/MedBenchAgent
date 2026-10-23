"""Experiment UI records and run locations; Compilation owns all execution state."""
import json
import re
from pathlib import Path
from uuid import uuid4

import tomlkit
from filelock import FileLock
from medbenchagent.settings import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from medbenchagent.runtime.run_artifacts import atomic_write_json

EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


def experiment_path(name, root=EXPERIMENTS_ROOT):
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise ValueError("Use a lowercase experiment name with letters, numbers and hyphens")
    return Path(root).resolve() / name


def create_experiment(name, *, root=EXPERIMENTS_ROOT, template=None):
    folder = experiment_path(name, root)
    if folder.exists():
        raise ValueError("Experiment already exists")
    text = (template or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    tomlkit.parse(text)
    folder.mkdir(parents=True)
    (folder / "config.toml").write_text(text, encoding="utf-8")
    record = {"name": name, "runs": [], "messages": []}
    atomic_write_json(folder / "session.json", record)
    return record


def load_experiment(name, *, root=EXPERIMENTS_ROOT):
    record = json.loads((experiment_path(name, root) / "session.json").read_text(encoding="utf-8"))
    if set(record) != {"name", "runs", "messages"}:
        raise ValueError("This experiment uses an old execution format; create a new experiment")
    return record


def save_experiment(record, *, root=EXPERIMENTS_ROOT):
    atomic_write_json(experiment_path(record["name"], root) / "session.json", record)


def list_experiments(root=EXPERIMENTS_ROOT):
    return [p.parent.name for p in sorted(Path(root).glob("*/session.json"))]


def config_path(name, root=EXPERIMENTS_ROOT):
    return experiment_path(name, root) / "config.toml"


def add_run(name, goal, *, root=EXPERIMENTS_ROOT, **options):
    from medbenchagent.compilation.run import create_run
    folder = experiment_path(name, root)
    with FileLock(folder / "session.lock"):
        record = load_experiment(name, root=root)
        identifier = "run-" + uuid4().hex[:12]
        directory = folder / "runs" / identifier
        create_run(directory, compilation_id=identifier, goal=goal,
                   config_path=config_path(name, root), **options)
        run = {"id": identifier, "directory": str(directory), "goal": goal}
        record["runs"].append(run)
        record["messages"].append({"role": "user", "text": goal, "run_id": identifier})
        save_experiment(record, root=root)
    return run
