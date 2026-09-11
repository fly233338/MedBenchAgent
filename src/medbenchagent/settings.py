"""Immutable application settings loaded from TOML and explicit runtime overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"

_ENVIRONMENT_OVERRIDES = {
    "DEEPSEEK_MODEL": ("llm", "model"),
    "DEEPSEEK_BASE_URL": ("llm", "base_url"),
    "DEEPSEEK_TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "BGE_EMBEDDING_MODEL": ("rag", "embedding_model"),
    "BGE_RERANKER_MODEL": ("rag", "reranker_model"),
    "RAG_MIN_RERANK_SCORE": ("rag", "min_rerank_score"),
}


@dataclass(frozen=True)
class PathSettings:
    dataset_catalog: Path
    blueprint_template: Path
    materials_root: Path
    runs_root: Path
    dataset_cache_root: Path
    rag_cache_root: Path


@dataclass(frozen=True)
class LlmSettings:
    model: str
    base_url: str
    timeout_seconds: float
    max_retries: int
    default_temperature: float
    trace_name: str


@dataclass(frozen=True)
class RagSettings:
    embedding_model: Path | None
    reranker_model: Path | None
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


@dataclass(frozen=True)
class DatasetProfileSettings:
    max_top_values: int
    max_top_value_chars: int


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings
    harness: dict[str, int]
    data_processing: dict[str, Any]
    llm: LlmSettings
    rag: RagSettings
    generation: GenerationSettings
    dataset_profile: DatasetProfileSettings
    category_weights: tuple[tuple[str, float], ...]
    default_config_path: Path
    local_config_path: Path | None
    environment_overrides: tuple[str, ...]

    def snapshot(self, graph_overrides: dict[str, object] | None = None) -> dict[str, object]:
        """Return non-sensitive settings without API or LangSmith credentials."""
        normalized_overrides = graph_overrides or {}
        generation = {
            key: getattr(self.generation, key) for key in self.generation.__dataclass_fields__
        }
        for key in ("seed", "max_tasks_per_image", "question_temperature"):
            if key in normalized_overrides:
                generation[key] = normalized_overrides[key]
        return {
            "settings": {
                "harness": self.harness,
                "data_processing": self.data_processing,
                "paths": {key: str(getattr(self.paths, key)) for key in self.paths.__dataclass_fields__},
                "llm": {
                    key: getattr(self.llm, key)
                    for key in self.llm.__dataclass_fields__
                },
                "rag": {
                    key: (
                        str(value) if isinstance(value, Path) else value
                    )
                    for key in self.rag.__dataclass_fields__
                    for value in (getattr(self.rag, key),)
                },
                "generation": generation,
                "dataset_profile": {
                    key: getattr(self.dataset_profile, key)
                    for key in self.dataset_profile.__dataclass_fields__
                },
                "sample_allocation": {"category_weights": dict(self.category_weights)},
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


def load_project_env() -> None:
    """Load the optional project .env without overriding process-level variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_llm_api_key() -> str | None:
    """Read the optional API key in the sole configuration module."""
    return os.getenv("DEEPSEEK_API_KEY") or None


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




def _settings_from(values: dict[str, Any], default_path: Path, local_path: Path | None, env_keys: list[str]) -> AppSettings:
    paths = values["paths"]
    llm = values["llm"]
    rag = values["rag"]
    generation = values["generation"]
    profile = values["dataset_profile"]
    weights = values["sample_allocation"]["category_weights"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("sample_allocation.category_weights must be a non-empty table.")
    normalized_weights = tuple(
        (key, _number(value, f"sample_allocation.category_weights.{key}", minimum=0, maximum=float("inf")))
        for key, value in weights.items()
    )
    if sum(value for _, value in normalized_weights) <= 0:
        raise ValueError("sample_allocation.category_weights must sum to a positive value.")
    chunk_size = _integer(rag["chunk_size"], "rag.chunk_size", minimum=1)
    chunk_overlap = _integer(rag["chunk_overlap"], "rag.chunk_overlap", minimum=0)
    if chunk_overlap >= chunk_size:
        raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_size.")
    return AppSettings(
        harness={key: _integer(value, f"harness.{key}", minimum=1) for key, value in values["harness"].items()},
        data_processing={"image": _string(values["data_processing"]["image"], "data_processing.image"),
                         "timeout_seconds": _integer(values["data_processing"]["timeout_seconds"], "data_processing.timeout_seconds", minimum=1)},
        paths=PathSettings(
            **{key: _resolve_path(value, f"paths.{key}") for key, value in paths.items()}
        ),
        llm=LlmSettings(
            model=_string(llm["model"], "llm.model"),
            base_url=_string(llm["base_url"], "llm.base_url"),
            timeout_seconds=_number(llm["timeout_seconds"], "llm.timeout_seconds", minimum=0.000001, maximum=float("inf")),
            max_retries=_integer(llm["max_retries"], "llm.max_retries", minimum=0),
            default_temperature=_number(llm["default_temperature"], "llm.default_temperature", minimum=0, maximum=2),
            trace_name=_string(llm["trace_name"], "llm.trace_name"),
        ),
        rag=RagSettings(
            embedding_model=_resolve_path(rag["embedding_model"], "rag.embedding_model", allow_empty=True),
            reranker_model=_resolve_path(rag["reranker_model"], "rag.reranker_model", allow_empty=True),
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
        category_weights=normalized_weights,
        default_config_path=default_path,
        local_config_path=local_path,
        environment_overrides=tuple(env_keys),
    )


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load default TOML, one explicit TOML override, then non-empty environment overrides."""
    load_project_env()
    values = _read_toml(DEFAULT_CONFIG_PATH)
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    local_path = _resolve_path(str(raw_path), "MEDBENCH_CONFIG") if raw_path else None
    if local_path:
        _merge(values, _read_toml(local_path))
    env_keys: list[str] = []
    for environment_name, (section, key) in _ENVIRONMENT_OVERRIDES.items():
        value = os.getenv(environment_name)
        if value:
            values[section][key] = value
            env_keys.append(f"{section}.{key}")
    return _settings_from(values, DEFAULT_CONFIG_PATH, local_path, env_keys)


def get_settings() -> AppSettings:
    """Return process-cached settings; clear_settings_cache is intended for tests and reloads."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def clear_settings_cache() -> None:
    """Clear cached settings after a test or explicit runtime configuration change."""
    global _SETTINGS
    _SETTINGS = None
