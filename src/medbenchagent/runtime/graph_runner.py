"""Run one LangGraph thread without terminal or Textual dependencies."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from medbenchagent.settings import active_settings, load_settings, run_settings


EventKind = Literal["initialized", "progress", "artifact", "needs_input", "finished", "error"]
_ARTIFACT_KEYS = {"run_directory", "effective_config_path", "construction_events_path",
                  "working_blueprint_path", "raw_output_path", "manifest_path", "delivery_path"}


@dataclass(frozen=True)
class RunEvent:
    run_session_id: str
    kind: EventKind
    payload: dict[str, Any]


def enforce_offline_mode() -> None:
    for name, value in {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ANONYMIZED_TELEMETRY": "False",
    }.items():
        os.environ[name] = value


def _event_parts(event: object) -> tuple[tuple[str, ...], str, dict]:
    if isinstance(event, tuple) and len(event) == 3:
        namespace, mode, payload = event
        if (isinstance(namespace, tuple) and all(isinstance(part, str) for part in namespace)
                and mode in {"tasks", "updates", "custom"} and isinstance(payload, dict)):
            return namespace, mode, payload
    raise ValueError("Expected stream event (namespace, tasks|updates|custom, dictionary payload)")


def _question(value: object) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("question"), str):
        raise ValueError("Clarification interrupt requires a nonempty question")
    question = value["question"].strip()
    if not question:
        raise ValueError("Clarification interrupt requires a nonempty question")
    return question


def _interrupt(payload: dict) -> dict | None:
    from langgraph.types import Interrupt

    if "__interrupt__" not in payload:
        return None
    values = payload["__interrupt__"]
    if not isinstance(values, tuple) or not all(isinstance(item, Interrupt) for item in values):
        raise ValueError("Expected a tuple of LangGraph Interrupt objects")
    if not values:
        return None
    _question(values[0].value)
    return values[0].value


def _recursion_limit(values: object) -> int:
    context = values.get("context", {}) if isinstance(values, dict) else {}
    calls = context.get("limits", {}).get("max_calls", 60)
    base = max(1_000, 3 * calls + 10)
    profile = values.get("dataset_profile", {}) if isinstance(values, dict) else {}
    images = profile.get("total_images") if isinstance(profile, dict) else None
    if not isinstance(images, int) or isinstance(images, bool) or images < 0:
        return base
    return max(base, images * 4 + 100)


def _metadata(values: dict) -> dict:
    context = values.get("context", {})
    return {
        "run_id": values.get("run_id"),
        "run_directory": values.get("run_directory"),
        "construction_events_path": context.get("construction_events_path"),
        "effective_config_path": context.get("effective_config_path"),
        "catalog_snapshot_path": context.get("task_catalog_snapshot_path"),
        "bir_path": context.get("bir_path"),
        "bir_revision": context.get("bir_revision"),
        "bir_manifest_digest": context.get("bir_manifest_digest"),
        "bir_template_revision": context.get("bir_template_revision"),
        "bir_summary_ref": context.get("bir_summary_ref"),
        "source_decision_path": context.get("source_decision_path"),
        "selected_assets": {
            dataset_id: asset.get("entry", {})
            for dataset_id, asset in context.get("dataset_assets", {}).items()
        },
    }


def _artifact_paths(value: object) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in _ARTIFACT_KEYS and isinstance(nested, str):
                yield key, nested
            else:
                yield from _artifact_paths(nested)


class GraphRunner:
    def __init__(self, *, graph_factory=None) -> None:
        self.graph_factory = graph_factory
        self._control = None

    def request_stop(self) -> None:
        """Request a checkpointed stop from the UI thread without closing a live stream."""
        control = self._control
        if control is not None:
            control.request_drain("user_stop")

    def _graph(self, connection):
        from langgraph.checkpoint.sqlite import SqliteSaver

        factory = self.graph_factory
        if factory is None:
            from medbenchagent.workflow.graph import compile_benchmark_graph

            factory = compile_benchmark_graph
        return factory(checkpointer=SqliteSaver(connection))

    def inspect(self, run: dict) -> dict:
        """Read recovery state after a process disappeared, without executing the graph."""
        identifier = run.get("id", "_single")
        return self.inspect_many([{**run, "id": identifier}])[identifier]

    def inspect_many(self, runs: list[dict]) -> dict[str, dict]:
        """Compile once per checkpoint database when reconciling several stale runs."""
        grouped: dict[str | None, list[dict]] = {}
        for run in runs:
            grouped.setdefault(run.get("runs_root"), []).append(run)
        results = {}
        for runs_root, group in grouped.items():
            if not runs_root:
                for run in group:
                    results[run["id"]] = {"status": "interrupted", "summary": "Run checkpoint path is missing"}
                continue
            checkpoint_path = Path(runs_root) / "checkpoints.sqlite"
            if not checkpoint_path.is_file():
                for run in group:
                    results[run["id"]] = {"status": "interrupted", "summary": "Run checkpoint database is missing"}
                continue
            connection = None
            try:
                connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
                graph = self._graph(connection)
                for run in group:
                    try:
                        snapshot = graph.get_state({"configurable": {"thread_id": run["thread_id"]}})
                        results[run["id"]] = self._inspection_result(snapshot)
                    except Exception as exc:
                        results[run["id"]] = {"status": "interrupted",
                                              "summary": f"Could not inspect checkpoint: {exc}"}
            except Exception as exc:
                for run in group:
                    results[run["id"]] = {"status": "interrupted",
                                          "summary": f"Could not inspect checkpoint: {exc}"}
            finally:
                if connection is not None:
                    connection.close()
        return results

    @staticmethod
    def _inspection_result(snapshot) -> dict:
        if not snapshot.values:
            return {"status": "interrupted", "summary": "Run checkpoint is missing"}
        pending = [item for task in snapshot.tasks for item in task.interrupts]
        status = ("waiting_input" if pending else snapshot.values.get("status")
                  if not snapshot.next else "interrupted")
        if status not in {"waiting_input", "complete", "blocked", "failed"}:
            status = "interrupted"
        return {"status": status, "summary": snapshot.values.get("summary", ""),
                **_metadata(snapshot.values)}

    def execute(self, run: dict, experiment_name: str, experiments_root: Path, *,
                resume: bool = False, answer: str | None = None,
                settings: Any = None, stop_requested: Callable[[], bool] | None = None) -> Iterator[RunEvent]:
        """Start or resume a run; return at an interrupt so the caller can answer later."""
        from langgraph.types import Command
        from langgraph.runtime import RunControl
        from langgraph.errors import GraphDrained
        from medbenchagent.runtime.dependencies import verify_resume
        from medbenchagent.runtime.preflight import preflight_errors, rag_capability

        enforce_offline_mode()
        identifier = run["id"]
        known_directory = run.get("run_directory")
        connection = None
        control = RunControl()
        self._control = control
        try:
            def honor_stop() -> bool:
                if stop_requested is None or not stop_requested():
                    return False
                from medbenchagent.runtime.dataset_runtime import OpenHandsDatasetRuntime
                OpenHandsDatasetRuntime.cancel_active_workers()
                control.request_drain("user_stop")
                return True

            if resume:
                runs_root = Path(run["runs_root"])
                checkpoint_path = runs_root / "checkpoints.sqlite"
                if not checkpoint_path.is_file():
                    raise ValueError("Run checkpoint database is missing")
                connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
                graph = self._graph(connection)
                config = {"configurable": {"thread_id": run["thread_id"]}, "recursion_limit": 1_000}
                previous = graph.get_state(config)
                if not previous.values:
                    raise ValueError("Unknown resume identifier")
                context = previous.values.get("context", {})
                known_directory = previous.values.get("run_directory") or known_directory
                snapshot_path = context.get("effective_config_path")
                if not snapshot_path:
                    raise ValueError("Run checkpoint or frozen configuration is missing")
                settings_context = run_settings(snapshot_path)
            else:
                settings_context = active_settings(
                    settings or load_settings(experiments_root / experiment_name / "config.toml")
                )

            with settings_context as settings:
                config_limits = getattr(settings, "harness", {})
                if resume:
                    config["recursion_limit"] = _recursion_limit(previous.values)
                if resume:
                    if previous.values.get("status") == "complete" and not previous.next:
                        yield RunEvent(identifier, "finished", {
                            "status": "complete", "summary": previous.values.get("summary", ""),
                            **_metadata(previous.values),
                        })
                        return
                    verify_resume(context)
                errors = preflight_errors(settings)
                if errors:
                    yield RunEvent(identifier, "error", {
                        "status": "blocked", "summary": "Local preflight failed: " + "; ".join(errors),
                        "run_directory": known_directory,
                    })
                    return
                if resume:
                    rag = rag_capability(settings)
                    if not rag["available"]:
                        yield RunEvent(identifier, "progress", {
                            "text": "Optional RAG unavailable: " + "; ".join(rag["reasons"]),
                        })
                if connection is None:
                    settings.paths.runs_root.mkdir(parents=True, exist_ok=True)
                    connection = sqlite3.connect(
                        str(settings.paths.runs_root / "checkpoints.sqlite"), check_same_thread=False
                    )
                    graph = self._graph(connection)
                    config = {"configurable": {"thread_id": run["thread_id"]},
                              "recursion_limit": _recursion_limit({"context": {"limits": config_limits}})}
                if resume:
                    pending = [item for task in previous.tasks for item in task.interrupts]
                    if pending:
                        question = _question(pending[0].value)
                        if answer is None:
                            yield RunEvent(identifier, "needs_input", {"question": question})
                            return
                        if not answer.strip():
                            raise ValueError("Clarification answer cannot be empty")
                        next_input: object = Command(resume=answer)
                    else:
                        if answer is not None:
                            raise ValueError("This run is not waiting for an answer")
                        next_input = None
                else:
                    if answer is not None:
                        raise ValueError("A new run cannot accept a clarification answer")
                    next_input = {
                        "user_requirement": run["requirement"],
                        "experiment_name": experiment_name,
                        "experiments_root": str(experiments_root),
                    }
                seen_artifacts: set[tuple[str, str]] = set()
                while True:
                    interrupted = None
                    initializing = not resume and next_input is not None
                    honor_stop()
                    # Freeze configuration before honoring a drain; an input-only checkpoint cannot resume.
                    for event in graph.stream(next_input, config,
                                              stream_mode=["tasks", "updates", "custom"],
                                              subgraphs=True, control=None if initializing else control,
                                              interrupt_after=["initialize"] if initializing else None,
                                              durability="sync"):
                        honor_stop()
                        namespace, mode, payload = _event_parts(event)
                        if mode == "tasks":
                            name = payload.get("name")
                            if name and "generation_step" not in name:
                                status = ("start" if "input" in payload else "failed" if payload.get("error")
                                          else "waiting" if payload.get("interrupts") else "done")
                                yield RunEvent(identifier, "progress", {
                                    "text": f"[{status}] {' / '.join(namespace + (name,))}",
                                })
                        elif mode == "custom":
                            if payload.get("kind") == "generation_progress":
                                yield RunEvent(identifier, "progress", {"text": str(payload)})
                        else:
                            if not namespace:
                                initialized = payload.get("initialize")
                                if isinstance(initialized, dict) and initialized.get("run_directory"):
                                    known_directory = initialized["run_directory"]
                                    yield RunEvent(identifier, "initialized", {
                                        "run_id": initialized.get("run_id"),
                                        "run_directory": known_directory,
                                        "runs_root": str(settings.paths.runs_root),
                                        "construction_events_path": initialized.get("construction_events_path"),
                                        "effective_config_path": initialized.get("effective_config_path"),
                                        "catalog_snapshot_path": initialized.get("task_catalog_snapshot_path"),
                                        "bir_path": initialized.get("bir_path"),
                                        "bir_revision": initialized.get("bir_revision"),
                                        "bir_manifest_digest": initialized.get("bir_manifest_digest"),
                                        "bir_template_revision": initialized.get("bir_template_revision"),
                                    })
                                    frozen_path = initialized.get("effective_config_path")
                                    if frozen_path and Path(frozen_path).is_file():
                                        with run_settings(frozen_path) as frozen:
                                            rag = rag_capability(frozen)
                                            if not rag["available"]:
                                                yield RunEvent(identifier, "progress", {
                                                    "text": "Optional RAG unavailable: " + "; ".join(rag["reasons"]),
                                                })
                            for name, path in _artifact_paths(payload):
                                if (name, path) not in seen_artifacts:
                                    seen_artifacts.add((name, path))
                                    yield RunEvent(identifier, "artifact", {"name": name, "path": path})
                            execute = payload.get("execute")
                            observations = execute.get("observations", []) if isinstance(execute, dict) else []
                            if observations:
                                item = observations[-1]
                                milestone = item.get("milestone")
                                suffix = f" [{milestone.get('kind')}]" if isinstance(milestone, dict) and milestone.get("kind") else ""
                                yield RunEvent(identifier, "progress", {
                                    "text": f"[tool] {item.get('tool')}: {item.get('status')}{suffix} — {item.get('summary')}",
                                })
                            if not namespace:
                                interrupted = interrupted or _interrupt(payload)
                    snapshot = graph.get_state(config)
                    values = snapshot.values or {}
                    known_directory = values.get("run_directory") or known_directory
                    config["recursion_limit"] = _recursion_limit(values)
                    if snapshot.next and stop_requested is not None and stop_requested():
                        yield RunEvent(identifier, "error", {
                            "status": "interrupted", "summary": "Run interrupted; checkpoint preserved",
                            "run_directory": known_directory,
                        })
                        return
                    if interrupted:
                        yield RunEvent(identifier, "needs_input", {
                            "question": _question(interrupted), **_metadata(values),
                        })
                        return
                    if snapshot.next:
                        next_input = None
                        continue
                    if values.get("status") not in {"complete", "blocked", "failed"}:
                        raise ValueError("Graph ended without a valid terminal status")
                    yield RunEvent(identifier, "finished", {
                        "status": values["status"],
                        "summary": values.get("summary", ""), **_metadata(values),
                    })
                    return
        except (GraphDrained, KeyboardInterrupt, EOFError):
            yield RunEvent(identifier, "error", {
                "status": "interrupted", "summary": "Run interrupted; checkpoint preserved",
                "run_directory": known_directory, "exit_code": 130,
            })
        except Exception as exc:
            from medbenchagent.runtime.llm import ModelUnavailable
            from medbenchagent.runtime.model_services import ModelDiscoveryError
            from medbenchagent.data.execution import ExecutionUnavailable
            category = ("model_service" if isinstance(exc, (ModelUnavailable, ModelDiscoveryError)) else
                        "execution_environment" if isinstance(exc, (ExecutionUnavailable, OSError)) else "runtime")
            restart_required = "Code or runtime contract changed" in str(exc) or not known_directory
            yield RunEvent(identifier, "error", {
                "status": "blocked" if restart_required else "interrupted",
                "summary": f"{type(exc).__name__}: {exc}", "run_directory": known_directory,
                "error_category": category,
                "recovery_hint": ("Correct the configuration or input and submit a new run."
                                  if restart_required else
                                  "Checkpoint preserved. Resolve the reported failure before resuming; code/input changes require a new run."),
            })
        finally:
            self._control = None
            if connection is not None:
                connection.close()
