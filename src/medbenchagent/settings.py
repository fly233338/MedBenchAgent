"""Immutable application settings loaded from TOML and explicit runtime overrides."""

from __future__ import annotations

import os
import math
import tomllib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"

_ENVIRONMENT_OVERRIDES = {
    "RAG_MIN_RERANK_SCORE": ("rag", "min_rerank_score"),
    "MEDBENCH_LLM_MODEL": ("llm", "model"),
    "MEDBENCH_LLM_BASE_URL": ("llm", "base_url"),
    "MEDBENCH_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "MEDBENCH_LLM_MAX_RETRIES": ("llm", "max_retries"),
}


@dataclass(frozen=True)
class PathSettings:
    dataset_catalog: Path
    blueprint_template: Path
    materials_root: Path
    runs_root: Path
    prepared_datasets_root: Path


@dataclass(frozen=True)
class LlmSettings:
    model: str
    base_url: str
    timeout_seconds: float
    max_retries: int
    default_temperature: float


@dataclass(frozen=True)
class RagSettings:
    embedding_base_url: str
    embedding_model: str
    reranker_base_url: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int
    candidate_count: int
    result_limit: int
    min_rerank_score: float
    reranker_max_length: int


@dataclass(frozen=True)
class GenerationSettings:
    seed: int
    max_tasks_per_image: int
    question_temperature: float
    fetch_batch_size: int
    question_template_count: int
    max_total_tasks_per_image: int | None = None


@dataclass(frozen=True)
class DatasetProfileSettings:
    max_top_values: int
    max_top_value_chars: int


@dataclass(frozen=True)
class BenchmarkSettings:
    default_sample_count: int


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings
    harness: dict[str, int]
    data_processing: dict[str, Any]
    llm: LlmSettings
    rag: RagSettings
    generation: GenerationSettings
    dataset_profile: DatasetProfileSettings
    dataset_sources: tuple[dict[str, Any], ...]
    benchmark: BenchmarkSettings
    default_config_path: Path
    local_config_path: Path | None
    environment_overrides: tuple[str, ...]
    service_status: dict[str, str]

    def snapshot(self, graph_overrides: dict[str, object] | None = None) -> dict[str, object]:
        """Return non-sensitive settings without API or LangSmith credentials."""
        normalized_overrides = graph_overrides or {}
        generation = {
            key: getattr(self.generation, key) for key in self.generation.__dataclass_fields__
        }
        for key in ("seed", "max_tasks_per_image", "max_total_tasks_per_image", "question_temperature"):
            if key in normalized_overrides:
                generation[key] = normalized_overrides[key]
        return {
            "settings": {
                "dataset_sources": list(self.dataset_sources),
                "benchmark": {"default_sample_count": self.benchmark.default_sample_count},
                "harness": self.harness,
                "data_processing": self.data_processing,
                "paths": {key: str(getattr(self.paths, key)) for key in self.paths.__dataclass_fields__},
                "llm": {
                    key: getattr(self.llm, key)
                    for key in self.llm.__dataclass_fields__
                },
                "rag": {
                    key: (
                        str(value) if isinstance(value, Path) else "" if value is None else value
                    )
                    for key in self.rag.__dataclass_fields__
                    for value in (getattr(self.rag, key),)
                },
                "model_services": self.service_status,
                "generation": generation,
                "dataset_profile": {
                    key: getattr(self.dataset_profile, key)
                    for key in self.dataset_profile.__dataclass_fields__
                },
            },
            "config_sources": {
                "default": {"path": str(self.default_config_path)},
                "local": (
                    {"path": str(self.local_config_path)}
                    if self.local_config_path
                    else None
                ),
            },
            "environment_overrides": list(self.environment_overrides),
            "graph_overrides": normalized_overrides,
        }


_SETTINGS: AppSettings | None = None
_RUN_SETTINGS: ContextVar[AppSettings | None] = ContextVar("run_settings", default=None)


@contextmanager
def run_settings(path):
    """Use the saved run configuration, never reinterpret current external configuration."""
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    saved = _settings_from(snapshot["settings"], Path(path), None, [])
    token = _RUN_SETTINGS.set(saved)
    try:
        yield saved
    finally:
        _RUN_SETTINGS.reset(token)


@contextmanager
def active_settings(settings: AppSettings):
    """Use one explicit CLI configuration for this invocation."""
    token = _RUN_SETTINGS.set(settings)
    try:
        yield settings
    finally:
        _RUN_SETTINGS.reset(token)


def load_project_env() -> None:
    """Load the optional project .env without overriding process-level variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_llm_api_key() -> str:
    """Read the Chat credential without storing it in run settings."""
    return os.getenv("MEDBENCH_LLM_API_KEY") or "local-vllm"


def get_rag_api_key(kind: str) -> str:
    """Keep each retrieval credential separate and outside snapshots."""
    if kind not in {"embedding", "reranker"}:
        raise ValueError("Unknown RAG service")
    return os.getenv(f"MEDBENCH_{kind.upper()}_API_KEY") or "local-vllm"


def _merge(base: dict[str, Any], override: dict[str, Any], *, prefix: str = "") -> None:
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration key: {name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a table.")
            _merge(base[key], value, prefix=name)
        else:
            base[key] = value


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    try:
        with path.open("rb") as source:
            value = tomllib.load(source)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML configuration: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a table: {path}")
    return value


def _resolve_path(value: object, key: str, *, allow_empty: bool = False) -> Path | None:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    if not value.strip():
        if allow_empty:
            return None
        raise ValueError(f"{key} must be non-empty.")
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _string(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value.strip()


def _integer(value: object, key: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{key} must be an integer {qualifier}.")
    return value


def _number(value: object, key: str, *, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number between {minimum} and {maximum}.")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return result




def _dataset_sources(values: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, list):
        raise ValueError("dataset_sources must be an array of tables")
    if not values:
        return ()

    from medbenchagent.data.contracts import SourcesArguments
    from medbenchagent.data.dataset_catalog import validate_dataset_id

    entries, seen = [], set()
    for item in values:
        if not isinstance(item, dict) or set(item) != {"dataset_id", "name", "description", "sources"}:
            raise ValueError("dataset_sources entries require dataset_id, name, description and sources")
        identifier = validate_dataset_id(item["dataset_id"])
        if identifier in seen:
            raise ValueError(f"Duplicate dataset_sources dataset_id: {identifier}")
        seen.add(identifier)
        parsed = SourcesArguments.model_validate({"dataset_id": identifier, "sources": item["sources"]})
        sources = [source.model_dump() for source in parsed.sources]
        for source in sources:
            source["root"] = str(_resolve_path(source["root"], f"dataset_sources.{identifier}.{source['id']}.root"))
        entries.append({"dataset_id": identifier, "name": _string(item["name"], "dataset_sources.name"),
                        "description": _optional_string(item["description"], "dataset_sources.description"), "sources": sources})
    return tuple(entries)


def _settings_from(values: dict[str, Any], default_path: Path, local_path: Path | None, env_keys: list[str]) -> AppSettings:
    paths = values["paths"]
    llm = values["llm"]
    rag = values["rag"]
    generation = values["generation"]
    profile = values["dataset_profile"]

    chunk_size = _integer(rag["chunk_size"], "rag.chunk_size", minimum=1)
    chunk_overlap = _integer(rag["chunk_overlap"], "rag.chunk_overlap", minimum=0)
    if chunk_overlap >= chunk_size:
        raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_size.")
    backend = values["data_processing"]["backend"]
    if backend not in {"auto", "docker", "local"}:
        raise ValueError("data_processing.backend must be auto, docker, or local")
    worker_backend = values["data_processing"]["coding_worker_backend"]
    if worker_backend not in {"openhands", "legacy"}:
        raise ValueError("data_processing.coding_worker_backend must be openhands or legacy")
    workspace_mode = values["data_processing"]["source_workspace_mode"]
    if workspace_mode not in {"readonly_mount", "copy"}:
        raise ValueError("data_processing.source_workspace_mode must be readonly_mount or copy")
    return AppSettings(
        benchmark=BenchmarkSettings(default_sample_count=_integer(values["benchmark"]["default_sample_count"],
                                                                  "benchmark.default_sample_count", minimum=1)),
        dataset_sources=_dataset_sources(values["dataset_sources"]),
        harness={key: _integer(value, f"harness.{key}", minimum=1) for key, value in values["harness"].items()},
        data_processing={"backend": backend, "image": _string(values["data_processing"]["image"], "data_processing.image"),
                         "memory_limit": _string(values["data_processing"]["memory_limit"], "data_processing.memory_limit"),
                         "threads": _integer(values["data_processing"]["threads"], "data_processing.threads", minimum=1),
                         "container_memory": _string(values["data_processing"]["container_memory"], "data_processing.container_memory"),
                         "timeout_seconds": _integer(values["data_processing"]["timeout_seconds"], "data_processing.timeout_seconds", minimum=1),
                         "coding_worker_backend": worker_backend,
                         "source_workspace_mode": workspace_mode,
                         "coding_worker_image": _string(values["data_processing"]["coding_worker_image"], "data_processing.coding_worker_image"),
                         "coding_worker_max_iterations": _integer(values["data_processing"]["coding_worker_max_iterations"], "data_processing.coding_worker_max_iterations", minimum=1),
                         "coding_worker_timeout_seconds": _integer(values["data_processing"]["coding_worker_timeout_seconds"], "data_processing.coding_worker_timeout_seconds", minimum=1),
                         "coding_worker_max_revisions": _integer(values["data_processing"]["coding_worker_max_revisions"], "data_processing.coding_worker_max_revisions", minimum=1)},
        paths=PathSettings(
            **{key: _resolve_path(value, f"paths.{key}") for key, value in paths.items()}
        ),
        llm=LlmSettings(
            model=_optional_string(llm["model"], "llm.model"),
            base_url=_string(llm["base_url"], "llm.base_url"),
            timeout_seconds=_number(llm["timeout_seconds"], "llm.timeout_seconds", minimum=0.000001, maximum=float("inf")),
            max_retries=_integer(llm["max_retries"], "llm.max_retries", minimum=0),
            default_temperature=_number(llm["default_temperature"], "llm.default_temperature", minimum=0, maximum=2),
        ),
        rag=RagSettings(
            embedding_base_url=_optional_string(rag["embedding_base_url"], "rag.embedding_base_url"),
            embedding_model=_optional_string(rag["embedding_model"], "rag.embedding_model"),
            reranker_base_url=_optional_string(rag["reranker_base_url"], "rag.reranker_base_url"),
            reranker_model=_optional_string(rag["reranker_model"], "rag.reranker_model"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            candidate_count=_integer(rag["candidate_count"], "rag.candidate_count", minimum=1),
            result_limit=_integer(rag["result_limit"], "rag.result_limit", minimum=1),
            min_rerank_score=_number(rag["min_rerank_score"], "rag.min_rerank_score", minimum=0, maximum=1),
            reranker_max_length=_integer(rag["reranker_max_length"], "rag.reranker_max_length", minimum=1),
        ),
        generation=GenerationSettings(
            seed=_integer(generation["seed"], "generation.seed", minimum=0),
            max_tasks_per_image=_integer(generation["max_tasks_per_image"], "generation.max_tasks_per_image", minimum=1),
            max_total_tasks_per_image=(_integer(generation["max_total_tasks_per_image"], "generation.max_total_tasks_per_image", minimum=1)
                                       if generation.get("max_total_tasks_per_image") is not None else None),
            question_temperature=_number(generation["question_temperature"], "generation.question_temperature", minimum=0, maximum=2),
            fetch_batch_size=_integer(generation["fetch_batch_size"], "generation.fetch_batch_size", minimum=1),
            question_template_count=_integer(
                generation["question_template_count"],
                "generation.question_template_count",
                minimum=1,
            ),
        ),
        dataset_profile=DatasetProfileSettings(
            max_top_values=_integer(profile["max_top_values"], "dataset_profile.max_top_values", minimum=1),
            max_top_value_chars=_integer(
                profile["max_top_value_chars"], "dataset_profile.max_top_value_chars", minimum=1
            ),
        ),
        default_config_path=default_path,
        local_config_path=local_path,
        environment_overrides=tuple(env_keys),
        service_status=dict(values.get("model_services", {})),
    )


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load default TOML, one explicit TOML override, then non-empty environment overrides."""
    load_project_env()
    values = _read_toml(DEFAULT_CONFIG_PATH)
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    local_path = None
    if raw_path:
        local_path = (Path(raw_path).expanduser().resolve() if config_path is not None
                      else _resolve_path(str(raw_path), "MEDBENCH_CONFIG"))
    if local_path:
        override = _read_toml(local_path)
        # Existing experiment TOMLs may still contain the removed tracing label.
        if isinstance(override.get("llm"), dict):
            override["llm"].pop("trace_name", None)
        _merge(values, override)
    # Redirect the old default saved in existing experiment TOMLs.
    if str(values["paths"]["runs_root"]).replace("\\", "/").rstrip("/") == "benchmarks/runs":
        values["paths"]["runs_root"] = ".runtime"
    env_keys: list[str] = []
    overrides = {target: (name, os.environ[name]) for name, target in _ENVIRONMENT_OVERRIDES.items()
                 if os.getenv(name)}
    for (section, key), (environment_name, value) in overrides.items():
        if value:
            try:
                if (section, key) in {("llm", "timeout_seconds"), ("rag", "min_rerank_score")}:
                    value = float(value)
                    # The settings validator below remains authoritative for ranges.
                    if not math.isfinite(value):
                        raise ValueError("must be finite")
                elif (section, key) == ("llm", "max_retries"):
                    value = _integer(int(value), environment_name)
            except ValueError as exc:
                raise ValueError(f"Invalid environment variable {environment_name}: {exc}") from exc
            values[section][key] = value
            if f"{section}.{key}" not in env_keys:
                env_keys.append(f"{section}.{key}")
    try:
        return _settings_from(values, DEFAULT_CONFIG_PATH, local_path, env_keys)
    except ValueError as exc:
        names = [name for name, (section, key) in _ENVIRONMENT_OVERRIDES.items()
                 if os.getenv(name) and f"{section}.{key}" in str(exc)]
        if names:
            raise ValueError(f"Invalid environment variable {names[-1]}: {exc}") from exc
        raise


def get_settings() -> AppSettings:
    """Return process-cached settings; clear_settings_cache is intended for tests and reloads."""
    global _SETTINGS
    if active := _RUN_SETTINGS.get():
        return active
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def clear_settings_cache() -> None:
    """Clear cached settings after a test or explicit runtime configuration change."""
    global _SETTINGS
    _SETTINGS = None
