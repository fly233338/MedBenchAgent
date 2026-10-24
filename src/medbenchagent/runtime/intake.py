"""User drafts and deterministic input adaptation; no execution or semantic planning."""
import json
import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import tomlkit
from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.compilation.contracts import GoalSpec
from medbenchagent.compilation.run import required_roles
from medbenchagent.data.dataset import DatasetRef, DatasetSource, FileDatasetReader
from medbenchagent.data.task_data import KeyFile, SourceManifest
from medbenchagent.settings import (
    DEFAULT_CONFIG_PATH, get_coding_api_key, get_llm_api_key,
    load_coding_settings, load_supervisor_settings,
)


class SelectedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    purpose: str = "Selected key file"


class InputDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = ""
    source_dirs: list[str] = Field(default_factory=list)
    key_files: list[SelectedFile] = Field(default_factory=list)
    goal_kind: Literal["profile", "scenarios", "executable_leaves", "benchmark"] = "benchmark"
    sample_count: int | None = Field(default=None, gt=0)
    allow_limited_delivery: bool = True
    canonical_manifest: str | None = None
    source_manifest: str | None = None
    dataset_id: str | None = None
    language: str = "en"

    def goal_spec(self):
        plan = self.goal_kind in {"scenarios", "executable_leaves"}
        return GoalSpec(kind="plan" if plan else self.goal_kind,
                        plan_level=self.goal_kind if plan else None,
                        allow_limited_delivery=self.allow_limited_delivery)


def input_path(value: str) -> Path:
    return Path(value.strip().strip('"\'').strip()).expanduser().resolve()


def suggest_draft(text: str, previous: InputDraft | None = None) -> InputDraft:
    draft = previous.model_copy(deep=True) if previous else InputDraft()
    draft.goal = text.strip()
    candidates = re.findall(r"[\"'`]((?:[A-Za-z]:[\\/]|/)[^\"'`]+)[\"'`]", text)
    for line in text.splitlines():
        stripped = line.strip().strip('"\'`')
        if stripped and Path(stripped).expanduser().is_dir():
            candidates.append(stripped)
    candidates += re.findall(r"(?<!\S)(?:[A-Za-z]:[\\/]|/)[^\s\"'`<>]+", text)
    paths = []
    for value in candidates:
        value = value.rstrip(".,;!?，。；！？）)]}")
        if value and value not in paths and not any(value != p and p.startswith(value) for p in candidates):
            paths.append(value)
    if paths:
        draft.source_dirs = paths
        draft.canonical_manifest = None
    count = re.search(r"(?<!\d)([1-9]\d*)\s*(?:道?题|questions?\b|items?\b)", text, re.IGNORECASE)
    if count:
        draft.sample_count = int(count.group(1))
    return draft


def validate_draft(draft: InputDraft, output: Path):
    if not draft.goal.strip():
        raise ValueError("Goal required.")
    goal = draft.goal_spec()
    required_roles(goal, raw_source=not bool(draft.canonical_manifest))
    if draft.goal_kind != "benchmark" and draft.sample_count is not None:
        raise ValueError("Count applies to Benchmark only.")
    if draft.canonical_manifest:
        if draft.source_dirs or draft.key_files:
            raise ValueError("Choose canonical input or raw sources.")
        path = input_path(draft.canonical_manifest)
        FileDatasetReader(path)
        return {"dataset_manifest": path}
    if not draft.source_dirs:
        raise ValueError("Add a source directory.")
    roots = list(dict.fromkeys(input_path(value) for value in draft.source_dirs))
    for path in roots:
        if output.resolve().is_relative_to(path):
            raise ValueError("Source directory contains the Run output.")
        try:
            with os.scandir(path) as entries:
                next(entries, None)
        except OSError as exc:
            raise ValueError(f"Source not readable: {path}") from exc
    original = None
    if draft.source_manifest:
        original_path = input_path(draft.source_manifest)
        original = SourceManifest.model_validate_json(original_path.read_bytes())
    existing_ids = {(original_path.parent / item.root).resolve(): item.source_id for item in original.sources} if original else {}
    used = set(existing_ids.values())
    sources = []
    for index, path in enumerate(roots):
        identifier = existing_ids.get(path)
        if identifier is None:
            number = index + 1
            while f"source-{number}" in used:
                number += 1
            identifier = f"source-{number}"
        used.add(identifier)
        sources.append(DatasetSource(source_id=identifier, root=str(path)))
    files = []
    for selected in draft.key_files:
        path = input_path(selected.path)
        matching = [source for source in sources if path.is_relative_to(Path(source.root))]
        if not matching:
            raise ValueError(f"Key file is outside selected sources: {path}")
        source = max(matching, key=lambda entry: len(Path(entry.root).parts))
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            raise ValueError(f"Key file not readable: {path}") from exc
        files.append(KeyFile(source_id=source.source_id, path=path.relative_to(source.root).as_posix(), purpose=selected.purpose))
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", roots[0].name).strip("-_") or "dataset"
    definition = {"sources": [s.model_dump() for s in sources], "key_files": [f.model_dump() for f in files]}
    revision = "input-" + hashlib.sha256(json.dumps(definition, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    dataset = DatasetRef(dataset_id=draft.dataset_id or name, revision=revision)
    if original is not None:
        previous = {"sources": [{"source_id": s.source_id, "root": str((original_path.parent / s.root).resolve())} for s in original.sources],
                    "key_files": [f.model_dump() for f in original.key_files]}
        if previous == definition and dataset.dataset_id == original.dataset.dataset_id:
            dataset = original.dataset
    manifest = SourceManifest(dataset=dataset,
                              sources=sources, key_files=files)
    return {"source": manifest}


def prepare_input(draft: InputDraft, directory: Path):
    result = validate_draft(draft, directory)
    if "source" in result:
        result["source_manifest"] = snapshot_source(result.pop("source"), directory)
    return {**result, "goal_spec": draft.goal_spec(), "sample_count": draft.sample_count,
            "language": draft.language if draft.goal_kind == "benchmark" else None}


def snapshot_source(source: SourceManifest, directory: Path) -> Path:
    from medbenchagent.data.task_data import write_once
    content = (source.model_dump_json(indent=2) + "\n").encode()
    # Dataset revisions are semantic identifiers, never filesystem names.
    path = directory / "inputs" / "sources" / (hashlib.sha256(content).hexdigest() + ".json")
    write_once(path, content)
    return path


def register_draft(directory, draft, *, catalog=None):
    from medbenchagent.data.dataset_catalog import register_dataset
    result = validate_draft(draft.model_copy(update={"goal": draft.goal or "Register source"}), Path(directory))
    if "source" in result:
        source = result["source"]
        return register_dataset(source.dataset.dataset_id, snapshot_source(source, Path(directory)), kind="source", catalog=catalog)
    path = result["dataset_manifest"]
    return register_dataset(FileDatasetReader(path).get_manifest().dataset_id, path, catalog=catalog)


def select_dataset(dataset_id, draft, *, catalog=None):
    from medbenchagent.data.dataset_catalog import registered_input
    selected = registered_input(dataset_id, **({"catalog": catalog} if catalog else {}))
    result = draft.model_copy(deep=True)
    if "dataset_manifest" in selected:
        result.canonical_manifest = str(selected["dataset_manifest"])
        result.source_manifest, result.source_dirs, result.key_files = None, [], []
    else:
        path = selected["source_manifest"]
        source = SourceManifest.model_validate_json(path.read_bytes())
        roots = {s.source_id: (path.parent / s.root).resolve() for s in source.sources}
        result.source_dirs = [str(root) for root in roots.values()]
        result.key_files = [SelectedFile(path=str(roots[f.source_id] / f.path), purpose=f.purpose) for f in source.key_files]
        result.source_manifest, result.canonical_manifest = str(path), None
    result.dataset_id = dataset_id
    return result


def connection_values(config: Path) -> dict:
    if (config.parent / "run.json").exists():
        from medbenchagent.compilation.run import RunOptions
        frozen = RunOptions.model_validate_json((config.parent / "run.json").read_bytes())
        common = frozen.supervisor_settings["llm"]
        coding = frozen.coding_settings or {"llm": common, "coding": {}}
        runtime = coding["coding"]
        return {"base_url": common["base_url"], "model": common["model"],
                "coding_base_url": coding["llm"]["base_url"], "coding_model": coding["llm"]["model"],
                "worker_venv": str(Path(runtime["worker_python"]).parent.parent) if runtime.get("worker_python") else "",
                "skills": runtime.get("skill_names", [])}
    values = tomlkit.parse(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    override = tomlkit.parse(config.read_text(encoding="utf-8"))
    def merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                merge(base[key], value)
            else:
                base[key] = value
    merge(values, override)
    return {"base_url": str(values["llm"]["base_url"]), "model": str(values["llm"]["model"]),
            "coding_base_url": str(values["coding"].get("llm", {}).get("base_url", "")),
            "coding_model": str(values["coding"].get("llm", {}).get("model", "")),
            "worker_venv": str(values["coding"]["worker_venv"]),
            "skills": list(values["coding"]["skills"]["enabled"])}


def save_connection(config: Path, values: dict):
    if (config.parent / "run.json").exists():
        raise ValueError("Run frozen. Clone to edit.")
    for value in (values["base_url"], values.get("coding_base_url", "")):
        if not value:
            continue
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use an HTTP(S) endpoint without credentials or query.")
    if not values["base_url"]:
        raise ValueError("Endpoint required.")
    document = tomlkit.parse(config.read_text(encoding="utf-8"))
    document.setdefault("llm", {}).update({key: values[key] for key in ("base_url", "model")})
    coding = document.setdefault("coding", {})
    coding["worker_venv"] = values["worker_venv"]
    coding["skills"] = {"enabled": values["skills"]}
    coding["llm"] = {key: values["coding_" + key] for key in ("base_url", "model") if values.get("coding_" + key)}
    config.write_text(tomlkit.dumps(document), encoding="utf-8")


def set_session_credentials(common="", coding=""):
    for name, value in (("MEDBENCH_LLM_API_KEY", common), ("MEDBENCH_CODING_LLM_API_KEY", coding)):
        if value.strip():
            os.environ[name] = value.strip()


def save_advanced_config(config, text):
    from tempfile import TemporaryDirectory
    from medbenchagent.settings import _merge, load_work_agent_settings, load_planner_settings
    if (config.parent / "run.json").exists():
        raise ValueError("Run frozen. Clone to edit.")
    document = tomlkit.parse(text)
    checked = tomlkit.parse(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    _merge(checked, document)
    if document.get("paths") != tomlkit.parse(config.read_text(encoding="utf-8")).get("paths"):
        raise ValueError("Run paths are fixed.")
    for section in (checked["llm"], checked["coding"].get("llm", {})):
        if section.get("base_url"):
            url = urlsplit(section["base_url"])
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("Endpoint cannot contain credentials or query.")
    with TemporaryDirectory(prefix="medbench-config-") as folder:
        path = Path(folder) / "config.toml"
        path.write_text(text, encoding="utf-8")
        load_supervisor_settings(path)
        load_planner_settings(path)
        load_coding_settings(path)
        for role in ("data_processing", "generation", "review"):
            load_work_agent_settings(role, path)
    config.write_text(text, encoding="utf-8")


def _check_coding_environment(settings):
    if not Path(settings.worker_python).is_file():
        raise ValueError(f"Coding Python missing: {settings.worker_python}")
    code = "import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in ['openhands-sdk','openhands-tools']}))"
    result = subprocess.run([settings.worker_python, "-c", code], capture_output=True, text=True,
                            encoding="utf-8", timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode or json.loads(result.stdout) != settings.sdk_packages:
        raise ValueError("Coding SDK mismatch. Install coding-worker.requirements.txt.")


def check_connections(config: Path, draft: InputDraft):
    from medbenchagent.runtime.model_services import resolve_model_service
    roles = required_roles(draft.goal_spec(), raw_source=not bool(draft.canonical_manifest))
    common = load_supervisor_settings(config)
    resolved = {"model": resolve_model_service("Chat", common.llm.base_url, common.llm.model, get_llm_api_key())}
    if "coding" in roles:
        settings = load_coding_settings(config)
        _check_coding_environment(settings)
        resolved["coding_model"] = resolve_model_service("Coding", settings.llm.base_url, settings.llm.model, get_coding_api_key())
    return resolved


def probe_connections(config: Path, draft: InputDraft):
    from medbenchagent.runtime.model_services import available_models
    common = load_supervisor_settings(config)
    result = {"model": available_models("Chat", common.llm.base_url, get_llm_api_key())}
    if "coding" in required_roles(draft.goal_spec(), raw_source=not bool(draft.canonical_manifest)):
        coding = load_coding_settings(config)
        _check_coding_environment(coding)
        result["coding_model"] = available_models("Coding", coding.llm.base_url, get_coding_api_key())
    return result
