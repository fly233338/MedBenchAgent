"""Immutable application settings loaded from TOML and explicit runtime overrides."""

from __future__ import annotations

import os
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv


_SOURCE_LAYOUT = Path(__file__).resolve().parents[1].name == "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2] if _SOURCE_LAYOUT else Path.cwd()
_CONFIG_ROOT = PROJECT_ROOT / "config" if _SOURCE_LAYOUT else Path(__file__).parent / "_config"
DEFAULT_CONFIG_PATH = _CONFIG_ROOT / "default.toml"

_ENVIRONMENT_OVERRIDES = {
    "RAG_MIN_RERANK_SCORE": ("rag", "min_rerank_score"),
    "MEDBENCH_LLM_MODEL": ("llm", "model"),
    "MEDBENCH_LLM_BASE_URL": ("llm", "base_url"),
    "MEDBENCH_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "MEDBENCH_LLM_MAX_RETRIES": ("llm", "max_retries"),
    "MEDBENCH_LLM_REASONING_EFFORT": ("llm", "reasoning_effort"),
}


@dataclass(frozen=True)
class LlmSettings:
    model: str
    base_url: str
    timeout_seconds: float
    max_retries: int
    default_temperature: float
    reasoning_effort: str


@dataclass(frozen=True)
class AgentLoopSettings:
    """Native agent invocation limits; never cumulative compilation quotas."""

    max_model_calls: int = 24
    max_tool_calls: int = 64
    context_tokens: int = 24000
    max_output_tokens: int = 4096
    summary_trigger_tokens: int = 8000
    summary_keep_tokens: int = 2000

    @classmethod
    def parse(cls, values):
        if not isinstance(values, dict) or set(values) != set(cls.__dataclass_fields__):
            raise ValueError("Unexpected or missing agent loop settings")
        result = cls(
            **{key: _integer(value, f"loop.{key}", minimum=1) for key, value in values.items()}
        )
        if not (
            result.summary_keep_tokens
            < result.summary_trigger_tokens
            < result.context_tokens - result.max_output_tokens
        ):
            raise ValueError("Agent summary limits must leave room for context and output")
        return result


@dataclass(frozen=True)
class SupervisorSettings:
    """Only the model and context limits needed by independent supervision."""

    llm: LlmSettings
    context_chars: int
    artifact_chars: int
    max_artifacts: int
    history_items: int
    catalog_page_size: int
    max_lookups: int
    loop: AgentLoopSettings = field(default_factory=AgentLoopSettings)

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm": asdict(self.llm),
            "supervisor": {
                "context_chars": self.context_chars,
                "artifact_chars": self.artifact_chars,
                "max_artifacts": self.max_artifacts,
                "history_items": self.history_items,
                "catalog_page_size": self.catalog_page_size,
                "max_lookups": self.max_lookups,
                "loop": asdict(self.loop),
            },
        }

    @classmethod
    def from_snapshot(cls, values: dict[str, Any]) -> "SupervisorSettings":
        if not isinstance(values, dict) or set(values) != {"llm", "supervisor"}:
            raise ValueError("Supervisor settings require llm and supervisor sections")
        limits = values["supervisor"]
        if not isinstance(limits, dict) or set(limits) != {
            "context_chars",
            "artifact_chars",
            "max_artifacts",
            "history_items",
            "catalog_page_size",
            "max_lookups",
            "loop",
        }:
            raise ValueError("Unexpected or missing Supervisor limit fields")
        return cls(
            llm=_checked_llm_settings(values["llm"]),
            context_chars=_integer(limits["context_chars"], "supervisor.context_chars", minimum=1),
            artifact_chars=_integer(
                limits["artifact_chars"],
                "supervisor.artifact_chars",
                minimum=1,
            ),
            max_artifacts=_integer(limits["max_artifacts"], "supervisor.max_artifacts", minimum=1),
            history_items=_integer(limits["history_items"], "supervisor.history_items", minimum=1),
            catalog_page_size=_integer(
                limits["catalog_page_size"],
                "supervisor.catalog_page_size",
                minimum=1,
            ),
            max_lookups=_integer(limits["max_lookups"], "supervisor.max_lookups", minimum=1),
            loop=AgentLoopSettings.parse(limits["loop"]),
        )


@dataclass(frozen=True)
class TaskPlannerSettings:
    llm: LlmSettings
    context_chars: int
    page_size: int
    preview_chars: int
    max_scenarios: int
    loop: AgentLoopSettings

    def snapshot(self):
        values = asdict(self)
        return {"llm": values.pop("llm"), "planner": values}

    @classmethod
    def from_snapshot(cls, values):
        if not isinstance(values, dict) or set(values) != {"llm", "planner"}:
            raise ValueError("Planner settings require llm and planner sections")
        limits = values["planner"]
        if set(limits) != {"context_chars", "page_size", "preview_chars", "max_scenarios", "loop"}:
            raise ValueError("Unexpected or missing Planner settings")
        return cls(
            llm=_checked_llm_settings(values["llm"]),
            loop=AgentLoopSettings.parse(limits["loop"]),
            **{
                key: _integer(limits[key], f"planner.{key}", minimum=1)
                for key in ("context_chars", "page_size", "preview_chars", "max_scenarios")
            },
        )


@dataclass(frozen=True)
class WorkAgentSettings:
    llm: LlmSettings
    context_chars: int
    page_size: int
    preview_chars: int
    loop: AgentLoopSettings

    def snapshot(self):
        values = asdict(self)
        return {"llm": values.pop("llm"), "agent": values}

    @classmethod
    def from_snapshot(cls, values):
        if set(values) != {"llm", "agent"} or set(values["agent"]) != {
            "context_chars",
            "page_size",
            "preview_chars",
            "loop",
        }:
            raise ValueError("Unexpected native Agent configuration")
        limits = values["agent"]
        return cls(
            llm=_checked_llm_settings(values["llm"]),
            loop=AgentLoopSettings.parse(limits["loop"]),
            **{
                key: _integer(limits[key], key, minimum=1)
                for key in ("context_chars", "page_size", "preview_chars")
            },
        )


@dataclass(frozen=True)
class CodingSettings:
    """Frozen model and local execution settings, shared by this run's Workstreams."""

    llm: LlmSettings
    worker_python: str
    sdk_packages: dict[str, str]
    skill_names: list[str]
    max_iterations: int
    timeout_seconds: float
    max_output_tokens: int
    condenser_max_events: int
    condenser_max_tokens: int

    def snapshot(self) -> dict[str, Any]:
        values = asdict(self)
        return {"llm": values.pop("llm"), "coding": values}

    @classmethod
    def from_snapshot(cls, values: dict[str, Any]) -> "CodingSettings":
        if not isinstance(values, dict) or set(values) != {"llm", "coding"}:
            raise ValueError("Coding settings require llm and coding sections")
        runtime = values["coding"]
        if not isinstance(runtime, dict) or set(runtime) != set(cls.__dataclass_fields__) - {"llm"}:
            raise ValueError("Unexpected or missing Coding configuration fields")
        worker_python = _string(runtime["worker_python"], "coding.worker_python")
        if not Path(worker_python).is_absolute():
            raise ValueError("coding.worker_python must be an absolute path")
        packages = runtime["sdk_packages"]
        if not isinstance(packages, dict) or set(packages) != {"openhands-sdk", "openhands-tools"}:
            raise ValueError("Coding requires frozen OpenHands SDK and Tools versions")
        timeout = _number(
            runtime["timeout_seconds"],
            "coding.timeout_seconds",
            minimum=0.000001,
            maximum=float("inf"),
        )
        if not math.isfinite(timeout):
            raise ValueError("coding.timeout_seconds must be finite")
        limits = {
            name: _integer(runtime[name], f"coding.{name}", minimum=1)
            for name in (
                "max_iterations",
                "max_output_tokens",
                "condenser_max_events",
                "condenser_max_tokens",
            )
        }
        if limits["condenser_max_events"] <= 2:
            raise ValueError("coding.condenser_max_events must exceed the two retained events")
        names = runtime["skill_names"]
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError("coding.skills.enabled requires distinct skill names")
        return cls(
            llm=_checked_llm_settings(values["llm"]),
            worker_python=worker_python,
            sdk_packages={name: _string(value, name) for name, value in packages.items()},
            skill_names=names,
            timeout_seconds=timeout,
            **limits,
        )


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
class AppSettings:
    """Settings for standalone retrieval utilities; Agents load their own frozen settings."""

    llm: LlmSettings
    rag: RagSettings


def load_project_env() -> None:
    """Load the optional project .env without overriding process-level variables."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def local_config_path() -> Path | None:
    load_project_env()
    configured = os.getenv("MEDBENCH_CONFIG")
    path = (
        (PROJECT_ROOT / Path(configured).expanduser()).resolve()
        if configured
        else PROJECT_ROOT / "config/local.toml"
    )
    return path if path.is_file() else None


def load_paths(config_path: Path | None = None) -> dict[str, Path]:
    values = _read_toml(DEFAULT_CONFIG_PATH)["paths"]
    selected = config_path or local_config_path()
    if selected is not None:
        _merge(values, _read_toml(selected).get("paths", {}), prefix="paths")
    return {key: _resolve_path(value, "paths." + key) for key, value in values.items()}


def get_llm_api_key() -> str:
    """Read the Chat credential without storing it in run settings."""
    return os.getenv("MEDBENCH_LLM_API_KEY") or "local-vllm"


def get_coding_api_key() -> str:
    return os.getenv("MEDBENCH_CODING_LLM_API_KEY") or get_llm_api_key()


def get_rag_api_key(kind: str) -> str:
    """Keep each retrieval credential separate and outside snapshots."""
    if kind not in {"embedding", "reranker"}:
        raise ValueError("Unknown RAG service")
    return os.getenv(f"MEDBENCH_{kind.upper()}_API_KEY") or "local-vllm"


def _merge(base: dict[str, Any], override: dict[str, Any], *, prefix: str = "") -> None:
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            if prefix == "coding.llm" and key in LlmSettings.__dataclass_fields__:
                base[key] = value
                continue
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


def _llm_settings(llm: dict[str, Any]) -> LlmSettings:
    return LlmSettings(
        model=_optional_string(llm["model"], "llm.model"),
        base_url=_string(llm["base_url"], "llm.base_url"),
        timeout_seconds=_number(
            llm["timeout_seconds"],
            "llm.timeout_seconds",
            minimum=0.000001,
            maximum=float("inf"),
        ),
        max_retries=_integer(llm["max_retries"], "llm.max_retries", minimum=0),
        default_temperature=_number(
            llm["default_temperature"],
            "llm.default_temperature",
            minimum=0,
            maximum=2,
        ),
        reasoning_effort=_optional_string(llm["reasoning_effort"], "llm.reasoning_effort"),
    )


def _checked_llm_settings(values: dict[str, Any]) -> LlmSettings:
    if not isinstance(values, dict) or set(values) != set(LlmSettings.__dataclass_fields__):
        raise ValueError("Unexpected or missing LLM configuration fields")
    llm = _llm_settings(values)
    url = urlsplit(llm.base_url)
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError("llm.base_url must be an HTTP(S) base URL without credentials or query")
    return llm


def _apply_environment(values: dict[str, Any]) -> list[str]:
    """Apply the established overrides only to sections requested by the caller."""
    env_keys = []
    for name, (section, key) in _ENVIRONMENT_OVERRIDES.items():
        if section not in values or not os.getenv(name):
            continue
        value: Any = os.environ[name]
        try:
            if (section, key) in {("llm", "timeout_seconds"), ("rag", "min_rerank_score")}:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("must be finite")
            elif (section, key) == ("llm", "max_retries"):
                value = _integer(int(value), name)
        except ValueError as exc:
            raise ValueError(f"Invalid environment variable {name}: {exc}") from exc
        values[section][key] = value
        env_keys.append(f"{section}.{key}")
    return env_keys


def load_supervisor_settings(config_path: Path | None = None) -> SupervisorSettings:
    """Read only llm/supervisor values; do not parse datasets or preflight other services."""
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {name: defaults[name] for name in ("llm", "supervisor")}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        path = (
            Path(raw_path).expanduser().resolve()
            if config_path is not None
            else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        )
        override = _read_toml(path)
        _merge(values, {name: override[name] for name in values if name in override})
    _apply_environment(values)
    return SupervisorSettings.from_snapshot(values)


def load_planner_settings(config_path: Path | None = None) -> TaskPlannerSettings:
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {name: defaults[name] for name in ("llm", "planner")}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        path = (
            Path(raw_path).expanduser().resolve()
            if config_path is not None
            else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        )
        override = _read_toml(path)
        _merge(values, {name: override[name] for name in values if name in override})
    _apply_environment(values)
    return TaskPlannerSettings.from_snapshot(values)


def load_work_agent_settings(role: str, config_path: Path | None = None) -> WorkAgentSettings:
    section = {
        "data_processing": "data_processing_agent",
        "generation": "generation_agent",
        "review": "review_agent",
    }[role]
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {"llm": defaults["llm"], section: defaults[section]}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        path = (
            Path(raw_path).expanduser().resolve()
            if config_path is not None
            else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        )
        override = _read_toml(path)
        _merge(values, {name: override[name] for name in values if name in override})
    _apply_environment(values)
    return WorkAgentSettings.from_snapshot({"llm": values["llm"], "agent": values[section]})


def load_compilation_target(config_path: Path | None = None, *, sample_count: int | None = None):
    load_project_env()
    values = {"benchmark": _read_toml(DEFAULT_CONFIG_PATH)["benchmark"]}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        path = (
            Path(raw_path).expanduser().resolve()
            if config_path is not None
            else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        )
        override = _read_toml(path)
        if "benchmark" in override:
            _merge(values, {"benchmark": override["benchmark"]})
    return {
        "sample_count": _integer(
            sample_count
            if sample_count is not None
            else values["benchmark"]["default_sample_count"],
            "sample_count",
            minimum=1,
        ),
        "language": _string(values["benchmark"]["default_language"], "language"),
    }


def load_coding_settings(config_path: Path | None = None) -> CodingSettings:
    """Resolve only the local Coding environment and model; never provision dependencies."""
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {name: defaults[name] for name in ("llm", "coding")}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    overrides = {}
    if raw_path:
        path = (
            Path(raw_path).expanduser().resolve()
            if config_path is not None
            else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        )
        override = _read_toml(path)
        coding = override.get("coding", {})
        if not isinstance(coding, dict):
            raise ValueError("coding must be a table")
        overrides = coding.get("llm", {})
        if not isinstance(overrides, dict):
            raise ValueError("coding.llm must be a table")
        _merge(
            values,
            {
                **({"llm": override["llm"]} if "llm" in override else {}),
                "coding": {key: value for key, value in coding.items() if key != "llm"},
            },
        )
    _apply_environment(values)
    _merge(values["llm"], overrides, prefix="coding.llm")
    runtime = dict(values["coding"])
    runtime.pop("llm")
    runtime["skill_names"] = runtime.pop("skills")["enabled"]
    venv = _resolve_path(runtime.pop("worker_venv"), "coding.worker_venv")
    runtime["worker_python"] = str(
        venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    requirements = _CONFIG_ROOT / "coding-worker.requirements.txt"
    pins = dict(
        line.strip().split("==", 1)
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.startswith(("openhands-sdk==", "openhands-tools=="))
    )
    runtime["sdk_packages"] = pins
    return CodingSettings.from_snapshot({"llm": values["llm"], "coding": runtime})


def load_settings(config_path: Path | None = None) -> AppSettings:
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {key: defaults[key] for key in ("llm", "rag")}
    raw_path = config_path or os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        override = _read_toml(Path(raw_path))
        _merge(values, {key: override[key] for key in values if key in override})
    _apply_environment(values)
    rag = RagSettings(**values["rag"])
    if not 0 <= rag.chunk_overlap < rag.chunk_size:
        raise ValueError("RAG overlap must be smaller than the chunk size")
    return AppSettings(_checked_llm_settings(values["llm"]), rag)


def get_settings():
    return load_settings()
