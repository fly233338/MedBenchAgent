"""Command-line entry points for local benchmark construction and finalization."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class SessionQuit(Exception):
    """The user closed an input prompt while preserving its checkpoint."""

_NODE_LABELS = {
    "SupervisorAgent": "coordinate benchmark construction",
    "TaskPlanningAgent": "plan dataset and tasks",
    "DataProcessingAgent": "prepare raw data",
    "AllocationAgent": "allocate benchmark samples",
    "GenerationAgent": "generate benchmark",
    "FinalizationAgent": "validate and deliver benchmark",
    "generation_initialize": "initialize generation",
    "generation_finalize": "finalize generation",
}

_ARTIFACT_PATH_KEYS = {
    "working_blueprint_path",
    "effective_config_path",
    "construction_events_path",
    "raw_output_path",
    "manifest_path",
    "delivery_path",
}

_AUDIT_ARTIFACT_KEYS = {
    "audited",
    "quarantine",
    "report",
    "semantic_records",
}


def _enforce_offline_mode() -> None:
    """Disable optional outbound tracing, model downloads, and telemetry."""
    for name, value in {
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "ANONYMIZED_TELEMETRY": "False",
    }.items():
        os.environ[name] = value


def _read_nonempty(prompt: str, input_fn: InputFunction, output: OutputFunction) -> str:
    while True:
        value = input_fn(prompt).strip()
        if value:
            return value
        output("Input cannot be empty. Please try again.")


def _read_clarification(prompt: str, input_fn: InputFunction, output: OutputFunction,
                        settings: Any, graph: Any, config: dict[str, Any]) -> str:
    while True:
        value = _read_nonempty(prompt, input_fn, output)
        if value == "/quit":
            raise SessionQuit
        if value == "/config":
            snapshot = graph.get_state(config).values.get("context", {}).get("effective_config_path")
            _print_settings(settings, output, snapshot_path=snapshot)
        elif value == "/status":
            snapshot = graph.get_state(config)
            output(f"Run status: {snapshot.values.get('status', 'running')}")
        elif value.startswith("/"):
            output("While answering a question, use /status, /config or /quit.")
        else:
            return value


def _preflight_errors(settings: Any, *, build=True) -> list[str]:
    """Use the same intent-aware checks as the graph; RAG is optional."""
    from medbenchagent.runtime.preflight import preflight_errors
    return preflight_errors(settings, build=build)


def _report_preflight(settings, output, *, build):
    errors = _preflight_errors(settings, build=build)
    if errors:
        output("Local preflight failed:")
        for error in errors:
            output(f"- {error}")
        return False
    from medbenchagent.runtime.preflight import rag_capability
    rag = rag_capability(settings)
    if not rag["available"]:
        output("Optional RAG unavailable: " + "; ".join(rag["reasons"]))
    return True


def _print_json(value: object, output: OutputFunction) -> None:
    output(json.dumps(value, ensure_ascii=False, indent=2))


def _interrupt_question(payload: dict[str, Any]) -> str:
    question = payload.get("question") if isinstance(payload, dict) else None
    if isinstance(question, str) and question.strip():
        return question.strip()
    raise ValueError("Clarification interrupt requires a nonempty question")


def _render_interrupt(payload: dict[str, Any], output: OutputFunction) -> None:
    question = _interrupt_question(payload)
    output("")
    output(f"=== {question} ===")
    details = {key: value for key, value in payload.items() if key not in {"kind", "question"}}
    if details:
        _print_json(details, output)


def _interrupt_payload(value: object) -> dict[str, Any] | None:
    from langgraph.types import Interrupt

    if not isinstance(value, dict):
        raise ValueError("Updates payload must be a dictionary")
    if "__interrupt__" not in value:
        return None
    interrupts = value["__interrupt__"]
    if not isinstance(interrupts, tuple) or not all(isinstance(item, Interrupt) for item in interrupts):
        raise ValueError("Expected a tuple of LangGraph Interrupt objects")
    if not interrupts:
        return None
    payload = interrupts[0].value
    _interrupt_question(payload)
    return payload


def _event_parts(event: object) -> tuple[tuple[str, ...], str, object]:
    if isinstance(event, tuple) and len(event) == 3:
        namespace, mode, payload = event
        if (isinstance(namespace, tuple) and all(isinstance(part, str) for part in namespace)
                and isinstance(mode, str) and mode in {"tasks", "updates", "custom"}
                and isinstance(payload, dict)):
            return namespace, mode, payload
    raise ValueError("Expected stream event (namespace, tasks|updates|custom, dictionary payload)")


def _render_artifact_paths(payload: object, output: OutputFunction) -> None:
    found: list[tuple[str, str]] = []

    def collect(value: object, *, in_audit_artifacts: bool = False) -> None:
        if not isinstance(value, dict):
            return
        for key, nested in value.items():
            is_audit_artifacts = key == "audit_artifacts"
            if isinstance(nested, str) and (
                key in _ARTIFACT_PATH_KEYS or (in_audit_artifacts and key in _AUDIT_ARTIFACT_KEYS)
            ):
                found.append((key, nested))
            elif isinstance(nested, dict):
                collect(nested, in_audit_artifacts=is_audit_artifacts)

    collect(payload)
    for key, path in dict.fromkeys(found):
        output(f"[artifact] {key}: {path}")


def _render_update(payload: object, output: OutputFunction) -> None:
    _render_artifact_paths(payload, output)
    if isinstance(payload, dict) and isinstance(payload.get("execute"), dict):
        observations = payload["execute"].get("observations")
        if isinstance(observations, list) and observations:
            latest = observations[-1]
            if isinstance(latest, dict):
                output(f"[tool] {latest.get('tool')}: {latest.get('status')} — {latest.get('summary')}")
    if interrupt := _interrupt_payload(payload):
        output(f"[waiting] {_interrupt_question(interrupt)}")


def _render_task(namespace: tuple[str, ...], payload: object, output: OutputFunction) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Tasks payload must be a dictionary")
    task_name = payload.get("name")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("Task event requires a name")
    if "input" in payload:
        status = "start"
    elif "result" in payload:
        status = "failed" if payload.get("error") else "waiting" if payload.get("interrupts") else "done"
    else:
        raise ValueError("Task event requires input or result")
    names = [part.split(":", 1)[0] for part in namespace] + [task_name]
    name = " / ".join(names)
    if any(part in name for part in ("generation_step",)):
        return
    output(f"[{status}] {_NODE_LABELS.get(task_name, name)}")


def _render_progress(payload: object, output: OutputFunction) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("kind") == "generation_progress":
        output(
            f"[generation] stage={payload.get('stage') or 'complete'} "
            f"cursor={payload.get('cursor')} "
            f"completed={payload.get('completed_counts', {})}/"
            f"target={payload.get('target_counts', {})} "
            f"rejections={payload.get('rejections', {})} "
            f"exhausted={payload.get('exhausted')}"
        )


def _recursion_limit(values: object) -> int:
    if not isinstance(values, dict):
        return 1_000
    profile = values.get("dataset_profile", {})
    total_images = profile.get("total_images") if isinstance(profile, dict) else None
    if not isinstance(total_images, int) or isinstance(total_images, bool) or total_images < 0:
        return 1_000
    return max(1_000, total_images * 4 + 100)


def _print_settings(settings: Any, output: OutputFunction, *, snapshot_path: str | None = None) -> None:
    sources = None
    override_keys = settings.environment_overrides
    if snapshot_path:
        saved = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        sources = saved.get("config_sources")
        override_keys = saved.get("environment_overrides", [])
    source = ((sources or {}).get("local") or (sources or {}).get("default") or
              {"path": str(settings.local_config_path or settings.default_config_path)})["path"]
    output(f"Active config: {source}" + (f" (saved: {snapshot_path})" if snapshot_path else ""))
    output("Environment overrides: " + (", ".join(override_keys) if override_keys else "none"))
    output("")
    output("=== Effective configuration ===")
    output(f"LLM: {settings.llm.model} @ {settings.llm.base_url}")
    output(
        "Generation: "
        f"seed={settings.generation.seed}, "
        f"max_tasks_per_image={settings.generation.max_tasks_per_image} (per format), "
        f"max_total_tasks_per_image={settings.generation.max_total_tasks_per_image}, "
        f"question_temperature={settings.generation.question_temperature}"
    )
    output(f"Embedding: {settings.rag.embedding_model} @ {settings.rag.embedding_base_url}")
    output(f"Reranker: {settings.rag.reranker_model} @ {settings.rag.reranker_base_url}")
    output(f"Dataset catalog: {settings.paths.dataset_catalog}")
    output(f"Medical materials: {settings.paths.materials_root}")
    output(f"Run artifacts root: {settings.paths.runs_root}")


def _print_result(values: dict[str, Any], output: OutputFunction) -> None:
    output("")
    output(f"=== Benchmark construction: {values.get('status', 'complete')} ===")
    output(values.get("summary", ""))
    output(f"Run ID: {values.get('run_id')}")
    run_directory = values.get("run_directory")
    if not run_directory and isinstance(values.get("manifest_path"), str):
        run_directory = str(Path(values["manifest_path"]).parent)
    output(f"Run directory: {run_directory}")
    if values.get("prepared_entries"):
        output(f"Prepared datasets: {values['prepared_entries']}")
    output(f"Total items: {values.get('item_count', 0)}")
    output(f"Task counts: {values.get('task_counts', {})}")
    output(f"Shortfalls: {values.get('generation_shortfalls', {})}")
    output(f"Raw sources: {values.get('raw_source_paths', {})}")
    output(f"Delivery: {values.get('delivery_path')}")
    artifacts = values.get("audit_artifacts", {})
    if isinstance(artifacts, dict):
        output(f"Audited JSONL: {artifacts.get('audited')}")
        output(f"Audit report: {artifacts.get('report')}")
        report_path = artifacts.get("report")
        if isinstance(report_path, str) and Path(report_path).is_file():
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            counts = report["counts"]
            output("Delivery counts: " + ", ".join(f"{key}={counts[key]}" for key in
                   ("requested", "allocated", "generated", "accepted", "final")))
            output("Losses: " + ", ".join(f"{key}={counts[key]}" for key in
                   ("planning_shortfall", "generation_shortfall", "audit_rejected", "assembly_removed", "total_shortfall")))

    output(f"Manifest: {values.get('manifest_path')}")


def _announce_run(values: object, output: OutputFunction) -> str | None:
    if not isinstance(values, dict):
        return None
    run_directory = values.get("run_directory")
    if not isinstance(run_directory, str) or not run_directory:
        return None
    run_id = values.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = Path(run_directory).name
    output("")
    output("=== Run initialized ===")
    output(f"Run ID (directory name): {run_id}")
    output(f"Run directory: {run_directory}")
    return run_directory


def run_interactive_build(
    *,
    input_fn: InputFunction = input,
    output: OutputFunction = print,
    settings: Any = None,
    graph: Any = None,
    resume: str | None = None,
    dataset_id: str | None = None,
    dataset_ids: list[str] | None = None,
    initial_requirement: str | None = None,
    show_settings: bool = True,
    thread_id: str | None = None,
    experiment_name: str | None = None,
    experiments_root: str | None = None,
    run_metadata_callback: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Run the benchmark graph as a local, terminal-driven conversation."""
    _enforce_offline_mode()
    config: dict[str, Any] | None = None
    checkpoint_connection = None
    announced_run_directory: str | None = None
    known_run_directory = None
    previous = None
    saved_settings = None
    try:
        requirement = (
            ""
            if resume
            else initial_requirement or _read_nonempty("Describe the dataset preparation or Medical VLM benchmark: ", input_fn, output)
        )
        if settings is None:
            from medbenchagent.settings import get_settings

            settings = get_settings()
        if show_settings and not resume:
            _print_settings(settings, output)
        if not resume and not _report_preflight(settings, output, build=not dataset_id):
            return 2

        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command

        if graph is None:
            from medbenchagent.workflow.graph import compile_benchmark_graph

            settings.paths.runs_root.mkdir(parents=True, exist_ok=True)
            checkpoint_connection = sqlite3.connect(
                str(settings.paths.runs_root / "checkpoints.sqlite"), check_same_thread=False
            )
            graph = compile_benchmark_graph(checkpointer=SqliteSaver(checkpoint_connection))
        config = {
            "configurable": {"thread_id": resume or thread_id or str(uuid.uuid4())},
            "recursion_limit": 1_000,
        }
        output(f"Run thread: {config['configurable']['thread_id']}")
        next_input: object = {"user_requirement": requirement, "prepare_dataset_id": dataset_id,
                              "requested_dataset_ids": dataset_ids,
                              "entry_intent": "prepare_dataset" if dataset_id else "build_benchmark",
                              "experiment_name": experiment_name, "experiments_root": experiments_root}
        if resume:
            if previous is None:
                previous = graph.get_state(config)
            known_run_directory = previous.values.get("run_directory") or known_run_directory
            if not previous.values:
                raise ValueError("Unknown resume identifier")
            if previous.values.get("context", {}).get("prepare_dataset_id") != dataset_id:
                raise ValueError("Resume with the original dataset ID")
            if dataset_ids is not None and dataset_ids != previous.values.get("context", {}).get("requested_dataset_ids"):
                raise ValueError("Resume preserves the original dataset selection; do not change --dataset")
            if previous.values.get("status") == "complete" and not previous.next:
                saved_path = previous.values.get("context", {}).get("effective_config_path")
                if saved_path and Path(saved_path).is_file():
                    saved_document = json.loads(Path(saved_path).read_text(encoding="utf-8"))
                    if "embedding_base_url" in saved_document.get("settings", {}).get("rag", {}):
                        from medbenchagent.settings import run_settings

                        with run_settings(saved_path) as completed_settings:
                            _print_settings(completed_settings, output, snapshot_path=saved_path)
                    else:
                        sources = saved_document.get("config_sources", {})
                        source = sources.get("local") or sources.get("default") or {}
                        output(f"Active config: {source.get('path', saved_path)} (saved: {saved_path})")
                _print_result(previous.values, output)
                _render_artifact_paths(previous.values, output)
                return 0
            from medbenchagent.runtime.dependencies import verify_resume
            verify_resume(previous.values["context"])
            from medbenchagent.settings import run_settings
            saved_context = previous.values["context"]
            with run_settings(saved_context["effective_config_path"]) as saved_settings:
                _print_settings(saved_settings, output,
                                snapshot_path=saved_context["effective_config_path"])
                if not _report_preflight(saved_settings, output,
                                         build=saved_context.get("run_intent", {}).get("intent") == "build_benchmark"):
                    return 2
            pending = [i for task in previous.tasks for i in task.interrupts]
            if pending:
                _render_interrupt(pending[0].value, output)
                next_input = Command(resume=_read_clarification(
                    "Your response: ", input_fn, output, saved_settings, graph, config))
            else:
                next_input = None

        while True:
            interrupt: dict[str, Any] | None = None
            for event in graph.stream(
                next_input,
                config,
                stream_mode=["tasks", "updates", "custom"],
                subgraphs=True,
            ):
                namespace, mode, payload = _event_parts(event)
                if mode == "custom":
                    _render_progress(payload, output)
                elif mode == "tasks":
                    _render_task(namespace, payload, output)
                else:
                    if not namespace:
                        initialized = payload.get("initialize")
                        if isinstance(initialized, dict) and initialized.get("run_directory"):
                            known_run_directory = initialized["run_directory"]
                            announced_run_directory = _announce_run(initialized, output)
                            if run_metadata_callback:
                                run_metadata_callback(initialized)
                    _render_update(payload, output)
                    interrupt = interrupt or _interrupt_payload(payload)

            snapshot = graph.get_state(config)
            known_run_directory = snapshot.values.get("run_directory") or known_run_directory
            current_run_directory = snapshot.values.get("run_directory")
            if (
                isinstance(current_run_directory, str)
                and current_run_directory != announced_run_directory
            ):
                announced_run_directory = _announce_run(snapshot.values, output)
            config["recursion_limit"] = _recursion_limit(snapshot.values)
            if interrupt is not None:
                _render_interrupt(interrupt, output)
                answer = _read_clarification("Your response: ", input_fn, output,
                                             saved_settings or settings, graph, config)
                next_input = Command(resume=answer)
                continue
            if snapshot.next:
                next_input = None
                continue
            _print_result(snapshot.values, output)
            return 0 if snapshot.values.get("status", "complete") == "complete" else 1
    except SessionQuit:
        output("Session closed. Run checkpoint preserved.")
        return 0
    except (EOFError, KeyboardInterrupt):
        output("")
        output("Run interrupted. Reopen the experiment to resume while code and source inputs remain unchanged.")
        if known_run_directory:
            output(f"Preserved run artifacts: {known_run_directory}")
        return 130
    except Exception as exc:
        output("")
        output(f"Benchmark construction failed: {type(exc).__name__}: {exc}")
        output("Files and checkpoints are preserved. Reopen the experiment to resume if code is unchanged.")
        if known_run_directory:
            output(f"Preserved run artifacts: {known_run_directory}")
        return 1
    finally:
        if checkpoint_connection is not None:
            checkpoint_connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medbenchagent")
    parser.add_argument("--config", type=Path, help="TOML template for newly created experiments.")
    subcommands = parser.add_subparsers(dest="command")
    register = subcommands.add_parser("register", help="Register an already prepared bundle and Card.")
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--bundle", required=True)
    refresh = subcommands.add_parser("refresh-card", help="Refresh dataset metadata without re-preparing data.")
    refresh.add_argument("--dataset-id", required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "refresh-card":
        from medbenchagent.data.dataset_catalog import refresh_card
        _print_json(refresh_card(args.dataset_id), print)
        return
    if args.command == "register":
        from medbenchagent.data.dataset_catalog import register_dataset
        _print_json(register_dataset(args.dataset_id, args.bundle), print)
        return
    if args.config and not args.config.is_file():
        parser.error(f"Configuration template does not exist: {args.config}")
    from medbenchagent.tui import MBAApp
    _enforce_offline_mode()
    MBAApp(config_template=args.config.resolve() if args.config else None).run()
