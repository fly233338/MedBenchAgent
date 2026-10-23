"""Immutable application settings loaded from TOML and explicit runtime overrides."""

from __future__ import annotations

import os
import math
import tomllib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.toml"

_ENVIRONMENT_OVERRIDES = {
    "RAG_MIN_RERANK_SCORE": ("rag", "min_rerank_score"),
    "MEDBENCH_LLM_MODEL": ("llm", "model"),
    "MEDBENCH_LLM_BASE_URL": ("llm", "base_url"),
    "MEDBENCH_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds"),
    "MEDBENCH_LLM_MAX_RETRIES": ("llm", "max_retries"),
    "MEDBENCH_LLM_REASONING_EFFORT": ("llm", "reasoning_effort"),
}


@dataclass(frozen=True)
class PathSettings:
    dataset_catalog: Path
    bir_template: Path
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
        result = cls(**{key: _integer(value, f"loop.{key}", minimum=1)
                       for key, value in values.items()})
        if not (result.summary_keep_tokens < result.summary_trigger_tokens
                < result.context_tokens - result.max_output_tokens):
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
        if (not isinstance(limits, dict)
                or set(limits) != {"context_chars", "artifact_chars", "max_artifacts", "history_items",
                                   "catalog_page_size", "max_lookups", "loop"}):
            raise ValueError("Unexpected or missing Supervisor limit fields")
        return cls(
            llm=_checked_llm_settings(values["llm"]),
            context_chars=_integer(limits["context_chars"], "supervisor.context_chars", minimum=1),
            artifact_chars=_integer(limits["artifact_chars"], "supervisor.artifact_chars", minimum=1),
            max_artifacts=_integer(limits["max_artifacts"], "supervisor.max_artifacts", minimum=1),
            history_items=_integer(limits["history_items"], "supervisor.history_items", minimum=1),
            catalog_page_size=_integer(limits["catalog_page_size"], "supervisor.catalog_page_size", minimum=1),
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
        return cls(llm=_checked_llm_settings(values["llm"]),
                   loop=AgentLoopSettings.parse(limits["loop"]),
                   **{key: _integer(limits[key], f"planner.{key}", minimum=1)
                      for key in ("context_chars", "page_size", "preview_chars", "max_scenarios")})


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
        if set(values) != {"llm", "agent"} or set(values["agent"]) != {"context_chars", "page_size", "preview_chars", "loop"}:
            raise ValueError("Unexpected native Agent configuration")
        limits = values["agent"]
        return cls(llm=_checked_llm_settings(values["llm"]), loop=AgentLoopSettings.parse(limits["loop"]),
                   **{key: _integer(limits[key], key, minimum=1) for key in ("context_chars", "page_size", "preview_chars")})


@dataclass(frozen=True)
class CodingSettings:
    """Frozen model and local execution settings, shared by this run's Workstreams."""

    llm: LlmSettings
    worker_python: str
    sdk_packages: dict[str, str]
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
        if (not isinstance(runtime, dict)
                or set(runtime) != set(cls.__dataclass_fields__) - {"llm"}):
            raise ValueError("Unexpected or missing Coding configuration fields")
        worker_python = _string(runtime["worker_python"], "coding.worker_python")
        if not Path(worker_python).is_absolute():
            raise ValueError("coding.worker_python must be an absolute path")
        packages = runtime["sdk_packages"]
        if (not isinstance(packages, dict)
                or set(packages) != {"openhands-sdk", "openhands-tools"}):
            raise ValueError("Coding requires frozen OpenHands SDK and Tools versions")
        timeout = _number(runtime["timeout_seconds"], "coding.timeout_seconds",
                          minimum=0.000001, maximum=float("inf"))
        if not math.isfinite(timeout):
            raise ValueError("coding.timeout_seconds must be finite")
        limits = {
            name: _integer(runtime[name], f"coding.{name}", minimum=1)
            for name in ("max_iterations", "max_output_tokens", "condenser_max_events",
                         "condenser_max_tokens")
        }
        if limits["condenser_max_events"] <= 2:
            raise ValueError("coding.condenser_max_events must exceed the two retained events")
        return cls(
            llm=_checked_llm_settings(values["llm"]), worker_python=worker_python,
            sdk_packages={name: _string(value, name) for name, value in packages.items()},
            timeout_seconds=timeout, **limits,
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
class GenerationSettings:
    seed: int
    max_tasks_per_image: int
    fetch_batch_size: int
    question_template_count: int
    max_total_tasks_per_image: int | None = None


@dataclass(frozen=True)
class BenchmarkSettings:
    default_sample_count: int
    input_policy: str
    default_language: str = "en"


@dataclass(frozen=True)
class PlanningSettings:
    max_calls: int = 1200
    context_chars: int = 64000


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings
    harness: dict[str, int]
    data_processing: dict[str, Any]
    llm: LlmSettings
    rag: RagSettings
    generation: GenerationSettings
    dataset_sources: tuple[dict[str, Any], ...]
    benchmark: BenchmarkSettings
    planning: PlanningSettings
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
        for key in ("seed", "max_tasks_per_image", "max_total_tasks_per_image"):
            if key in normalized_overrides:
                generation[key] = normalized_overrides[key]
        return {
            "settings": {
                "dataset_sources": list(self.dataset_sources),
                "benchmark": {
                    "default_sample_count": self.benchmark.default_sample_count,
                    "input_policy": self.benchmark.input_policy,
                    "default_language": self.benchmark.default_language,
                },
                "planning": {key: getattr(self.planning, key) for key in self.planning.__dataclass_fields__},
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
            llm["timeout_seconds"], "llm.timeout_seconds", minimum=0.000001, maximum=float("inf"),
        ),
        max_retries=_integer(llm["max_retries"], "llm.max_retries", minimum=0),
        default_temperature=_number(
            llm["default_temperature"], "llm.default_temperature", minimum=0, maximum=2,
        ),
        reasoning_effort=_optional_string(llm["reasoning_effort"], "llm.reasoning_effort"),
    )


def _checked_llm_settings(values: dict[str, Any]) -> LlmSettings:
    if not isinstance(values, dict) or set(values) != set(LlmSettings.__dataclass_fields__):
        raise ValueError("Unexpected or missing LLM configuration fields")
    llm = _llm_settings(values)
    url = urlsplit(llm.base_url)
    if (url.scheme not in {"http", "https"} or not url.netloc or url.username is not None
            or url.password is not None or url.query or url.fragment):
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
        path = (Path(raw_path).expanduser().resolve() if config_path is not None
                else _resolve_path(str(raw_path), "MEDBENCH_CONFIG"))
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
        path = (Path(raw_path).expanduser().resolve() if config_path is not None
                else _resolve_path(str(raw_path), "MEDBENCH_CONFIG"))
        override = _read_toml(path)
        _merge(values, {name: override[name] for name in values if name in override})
    _apply_environment(values)
    return TaskPlannerSettings.from_snapshot(values)


def load_work_agent_settings(role: str, config_path: Path | None = None) -> WorkAgentSettings:
    section = {"data_processing": "data_processing_agent", "generation": "generation_agent", "review": "review_agent"}[role]
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {"llm": defaults["llm"], section: defaults[section]}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        path = Path(raw_path).expanduser().resolve() if config_path is not None else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        override = _read_toml(path)
        _merge(values, {name: override[name] for name in values if name in override})
    _apply_environment(values)
    return WorkAgentSettings.from_snapshot({"llm": values["llm"], "agent": values[section]})


def load_compilation_target(config_path: Path | None = None, *, sample_count: int | None = None):
    load_project_env()
    values = {"benchmark": _read_toml(DEFAULT_CONFIG_PATH)["benchmark"]}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    if raw_path:
        path = Path(raw_path).expanduser().resolve() if config_path is not None else _resolve_path(str(raw_path), "MEDBENCH_CONFIG")
        override = _read_toml(path)
        if "benchmark" in override:
            _merge(values, {"benchmark": override["benchmark"]})
    return {"sample_count": _integer(sample_count if sample_count is not None else values["benchmark"]["default_sample_count"], "sample_count", minimum=1),
            "language": _string(values["benchmark"]["default_language"], "language")}


def load_coding_settings(config_path: Path | None = None) -> CodingSettings:
    """Resolve only the local Coding environment and model; never provision dependencies."""
    load_project_env()
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    values = {name: defaults[name] for name in ("llm", "coding")}
    raw_path = config_path if config_path is not None else os.getenv("MEDBENCH_CONFIG")
    overrides = {}
    if raw_path:
        path = (Path(raw_path).expanduser().resolve() if config_path is not None
                else _resolve_path(str(raw_path), "MEDBENCH_CONFIG"))
        override = _read_toml(path)
        coding = override.get("coding", {})
        if not isinstance(coding, dict):
            raise ValueError("coding must be a table")
        overrides = coding.get("llm", {})
        if not isinstance(overrides, dict):
            raise ValueError("coding.llm must be a table")
        _merge(values, {**({"llm": override["llm"]} if "llm" in override else {}),
                        "coding": {key: value for key, value in coding.items() if key != "llm"}})
    _apply_environment(values)
    _merge(values["llm"], overrides, prefix="coding.llm")
    runtime = dict(values["coding"])
    runtime.pop("llm")
    venv = _resolve_path(runtime.pop("worker_venv"), "coding.worker_venv")
    runtime["worker_python"] = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    requirements = PROJECT_ROOT / "config" / "coding-worker.requirements.txt"
    pins = dict(line.strip().split("==", 1) for line in requirements.read_text(encoding="utf-8").splitlines()
                if line.startswith(("openhands-sdk==", "openhands-tools==")))
    runtime["sdk_packages"] = pins
    return CodingSettings.from_snapshot({"llm": values["llm"], "coding": runtime})




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

    chunk_size = _integer(rag["chunk_size"], "rag.chunk_size", minimum=1)
    chunk_overlap = _integer(rag["chunk_overlap"], "rag.chunk_overlap", minimum=0)
    if chunk_overlap >= chunk_size:
        raise ValueError("rag.chunk_overlap must be smaller than rag.chunk_size.")
    backend = values["data_processing"]["backend"]
    if backend not in {"auto", "docker", "local"}:
        raise ValueError("data_processing.backend must be auto, docker, or local")
    question_template_count = _integer(
        generation["question_template_count"],
        "generation.question_template_count",
        minimum=1,
    )
    if not 3 <= question_template_count <= 5:
        raise ValueError("generation.question_template_count must be between 3 and 5")
    if values["benchmark"]["default_language"] not in {"en", "zh"}:
        raise ValueError("benchmark.default_language must be en or zh")
    return AppSettings(
        benchmark=BenchmarkSettings(
            default_language=_string(values["benchmark"]["default_language"],
                                     "benchmark.default_language"),
            default_sample_count=_integer(values["benchmark"]["default_sample_count"],
                                          "benchmark.default_sample_count", minimum=1),
            input_policy=_string(values["benchmark"].get("input_policy", "image-only"),
                                 "benchmark.input_policy"),
        ),
        planning=PlanningSettings(
            max_calls=_integer(values["planning"]["max_calls"], "planning.max_calls", minimum=1),
            context_chars=_integer(values["planning"]["context_chars"], "planning.context_chars", minimum=4096),
        ),
        dataset_sources=_dataset_sources(values["dataset_sources"]),
        harness={key: _integer(value, f"harness.{key}", minimum=1) for key, value in values["harness"].items()},
        data_processing={"backend": backend,
                         "coding_worker_image": _string(values["data_processing"]["coding_worker_image"], "data_processing.coding_worker_image"),
                         "coding_worker_venv": str(_resolve_path(
                             values["data_processing"].get("coding_worker_venv", ".venv-coding"),
                             "data_processing.coding_worker_venv",
                         )),
                         "preparation_attempt_limits": {
                             key: _integer(value, f"data_processing.preparation_attempt_limits.{key}", minimum=0 if key == "mapping_revision_cycles" else 1)
                             for key, value in {
                                 "understand": values["data_processing"].get("preparation_attempt_limits", {}).get("understand", 2),
                                 "prepare_text": values["data_processing"].get("preparation_attempt_limits", {}).get("prepare_text", 2),
                                 "prepare_image": values["data_processing"].get("preparation_attempt_limits", {}).get("prepare_image", 2),
                                 "mapping_revision_cycles": values["data_processing"].get("preparation_attempt_limits", {}).get("mapping_revision_cycles", 1),
                             }.items()
                         },
                         "preparation_stage_budgets": values["data_processing"].get("preparation_stage_budgets", {})},
        paths=PathSettings(
            **{key: _resolve_path(value, f"paths.{key}") for key, value in paths.items()}
        ),
        llm=_llm_settings(llm),
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
            fetch_batch_size=_integer(generation["fetch_batch_size"], "generation.fetch_batch_size", minimum=1),
            question_template_count=question_template_count,
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
        _merge(values, override)
    env_keys = _apply_environment(values)
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
