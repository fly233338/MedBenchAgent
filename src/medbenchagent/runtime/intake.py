"""User drafts and deterministic input adaptation; no execution or semantic planning."""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal
from uuid import uuid4

import tomlkit
from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.compilation.contracts import GoalSpec
from medbenchagent.compilation.run import required_roles
from medbenchagent.data.dataset import DatasetRef, DatasetSource, FileDatasetReader
from medbenchagent.data.task_data import KeyFile, SourceManifest
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import (
    DEFAULT_CONFIG_PATH, get_coding_api_key, get_llm_api_key,
    load_coding_settings, load_supervisor_settings,
)


class SelectedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    purpose: str = "用户选定的关键文件"


class InputDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = ""
    source_dirs: list[str] = Field(default_factory=list)
    key_files: list[SelectedFile] = Field(default_factory=list)
    goal_kind: Literal["profile", "scenarios", "executable_leaves", "benchmark"] = "benchmark"
    sample_count: int | None = Field(default=None, gt=0)
    allow_limited_delivery: bool = True
    canonical_manifest: str | None = None

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
        raise ValueError("请填写任务要求")
    goal = draft.goal_spec()
    required_roles(goal, raw_source=not bool(draft.canonical_manifest))
    if draft.goal_kind != "benchmark" and draft.sample_count is not None:
        raise ValueError("只有 Benchmark 目标需要题目数量")
    if draft.canonical_manifest:
        if draft.source_dirs or draft.key_files:
            raise ValueError("已有 Dataset 与原始目录不能混用")
        path = input_path(draft.canonical_manifest)
        FileDatasetReader(path)
        return {"dataset_manifest": path}
    if not draft.source_dirs:
        raise ValueError("请选择至少一个数据目录")
    roots = list(dict.fromkeys(input_path(value) for value in draft.source_dirs))
    for path in roots:
        if output.resolve().is_relative_to(path):
            raise ValueError("数据目录不能包含运行输出目录，请选择实际数据子目录")
        try:
            next(path.iterdir(), None)
        except OSError as exc:
            raise ValueError(f"无法读取数据目录：{path}") from exc
    sources = [DatasetSource(source_id=f"source-{index + 1}", root=str(path)) for index, path in enumerate(roots)]
    files = []
    for selected in draft.key_files:
        path = input_path(selected.path)
        matching = [source for source in sources if path.is_relative_to(Path(source.root))]
        if not matching:
            raise ValueError(f"请先添加关键文件所属的来源目录：{path}")
        source = max(matching, key=lambda entry: len(Path(entry.root).parts))
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            raise ValueError(f"无法读取关键文件：{path}") from exc
        files.append(KeyFile(source_id=source.source_id, path=path.relative_to(source.root).as_posix(), purpose=selected.purpose))
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", roots[0].name).strip("-_") or "dataset"
    manifest = SourceManifest(dataset=DatasetRef(dataset_id=name, revision="input-" + uuid4().hex[:12]),
                              sources=sources, key_files=files)
    return {"source": manifest}


def prepare_input(draft: InputDraft, directory: Path):
    result = validate_draft(draft, directory)
    if "source" in result:
        path = directory / "inputs" / "sources.json"
        atomic_write_json(path, result.pop("source").model_dump(mode="json"))
        result["source_manifest"] = path
    return {**result, "goal_spec": draft.goal_spec(), "sample_count": draft.sample_count}


def connection_values(config: Path) -> dict:
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


def check_connections(config: Path, draft: InputDraft):
    from medbenchagent.runtime.model_services import resolve_model_service
    roles = required_roles(draft.goal_spec(), raw_source=not bool(draft.canonical_manifest))
    common = load_supervisor_settings(config)
    resolved = {"model": resolve_model_service("Chat", common.llm.base_url, common.llm.model, get_llm_api_key())}
    if "coding" in roles:
        settings = load_coding_settings(config)
        if not Path(settings.worker_python).is_file():
            raise ValueError(f"Coding Python 不存在：{settings.worker_python}")
        code = "import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) for p in ['openhands-sdk','openhands-tools']}))"
        result = subprocess.run([settings.worker_python, "-c", code], capture_output=True, text=True,
                                encoding="utf-8", timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode or json.loads(result.stdout) != settings.sdk_packages:
            raise ValueError("Coding 环境缺少固定版本的 OpenHands SDK/Tools，请按 requirements 配置")
        resolved["coding_model"] = resolve_model_service("Coding", settings.llm.base_url, settings.llm.model, get_coding_api_key())
    return resolved
