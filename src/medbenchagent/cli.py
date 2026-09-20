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

_NODE_LABELS = {
    "SupervisorAgent": "coordinate benchmark construction",
    "TaskPlanningAgent": "plan dataset and tasks",
    "DataProcessingAgent": "prepare raw data",
    "AllocationAgent": "allocate benchmark samples",
    "GenerationAgent": "generate benchmark",
    "generation_initialize": "initialize generation",
    "generation_finalize": "finalize generation",
}

_ARTIFACT_PATH_KEYS = {
    "working_blueprint_path",
    "effective_config_path",
    "construction_events_path",
    "raw_output_path",
    "manifest_path",
}

_AUDIT_ARTIFACT_KEYS = {
    "audited",
    "flagged",
    "rejected",
    "records",
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


def _preflight_errors(settings: Any) -> list[str]:
    """Return all local-resource problems that would prevent a complete run."""
    from medbenchagent.rag.pdf import RagUnavailable, discover_medical_materials

    errors: list[str] = []
    if not settings.paths.blueprint_template.is_file():
        errors.append(f"Blueprint template does not exist: {settings.paths.blueprint_template}")

    for label, path in (
        ("Embedding model", settings.rag.embedding_model),
        ("Reranker model", settings.rag.reranker_model),
    ):
        if path is None:
            errors.append(f"{label} is not configured.")
        elif not path.is_dir():
            errors.append(f"{label} directory does not exist: {path}")

    try:
        discover_medical_materials(settings.paths.materials_root)
    except RagUnavailable as exc:
        errors.append(str(exc))

    return errors


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


def _print_settings(settings: Any, output: OutputFunction) -> None:
    output("")
    output("=== Effective configuration ===")
    output(f"LLM: {settings.llm.model} @ {settings.llm.base_url}")
    output(
        "Generation: "
        f"seed={settings.generation.seed}, "
        f"max_tasks_per_image={settings.generation.max_tasks_per_image}, "
        f"question_temperature={settings.generation.question_temperature}"
    )
    output(f"Embedding: {settings.rag.embedding_model}")
    output(f"Reranker: {settings.rag.reranker_model}")
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
    output(f"Raw JSONL: {values.get('raw_output_path')}")
    artifacts = values.get("audit_artifacts", {})
    if isinstance(artifacts, dict):
        output(f"Audited JSONL: {artifacts.get('audited')}")
        output(f"Audit report: {artifacts.get('report')}")
        report_path = artifacts.get("report")
        if isinstance(report_path, str) and Path(report_path).is_file():
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            output(
                "Audit summary: "
                f"accepted={report.get('hard_audit', {}).get('accepted', 0)}, "
                f"rejected={report.get('hard_audit', {}).get('rejected', 0)}, "
                f"semantic_flags={report.get('semantic_audit', {}).get('flagged', 0)}"
            )
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
    experiment: dict | None = None,
) -> int:
    """Run the benchmark graph as a local, terminal-driven conversation."""
    _enforce_offline_mode()
    config: dict[str, Any] | None = None
    checkpoint_connection = None
    announced_run_directory: str | None = None
    known_run_directory = experiment.get("run_directory") if experiment else None
    previous = None
    interrupted = False
    failed = False
    from medbenchagent.runtime.experiments import update_experiment, snapshot_updates
    try:
        requirement = (
            ""
            if resume or experiment
            else _read_nonempty("Describe the dataset preparation or Medical VLM benchmark: ", input_fn, output)
        )
        if settings is None:
            from medbenchagent.settings import get_settings

            settings = get_settings()
        _print_settings(settings, output)
        errors = [] if resume or dataset_id or experiment else _preflight_errors(settings)
        if errors:
            output("")
            output("Local preflight failed:")
            for error in errors:
                output(f"- {error}")
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
            "configurable": {"thread_id": experiment["thread_id"] if experiment else resume or str(uuid.uuid4())},
            "recursion_limit": 1_000,
        }
        output(f"Experiment: {experiment['name']}" if experiment else f"Resume identifier: {config['configurable']['thread_id']}")
        if experiment:
            previous = graph.get_state(config)
            known_run_directory = previous.values.get("run_directory") or known_run_directory
            dataset_id = experiment["dataset_id"]
            dataset_ids = experiment["requested_dataset_ids"]
            if previous.values:
                resume = experiment["thread_id"]
            elif experiment.get("run_id"):
                raise ValueError("Experiment checkpoint is missing; use the preserved run artifacts for recovery")
            else:
                requirement = _read_nonempty("Describe the dataset preparation or Medical VLM benchmark: ", input_fn, output)
        next_input: object = {"user_requirement": requirement, "prepare_dataset_id": dataset_id,
                              "requested_dataset_ids": dataset_ids,
                              "entry_intent": "prepare_dataset" if dataset_id else None if experiment else "build_benchmark"}
        if resume:
            if previous is None:
                previous = graph.get_state(config)
            known_run_directory = previous.values.get("run_directory") or known_run_directory
            if not previous.values:
                raise ValueError("Unknown resume identifier")
            if not experiment and previous.values.get("context", {}).get("prepare_dataset_id") != dataset_id:
                raise ValueError("Resume with the original build/prepare command and dataset ID")
            if dataset_ids is not None and dataset_ids != previous.values.get("context", {}).get("requested_dataset_ids"):
                raise ValueError("Resume preserves the original dataset selection; do not change --dataset")
            if previous.values.get("status") == "complete" and not previous.next:
                _print_result(previous.values, output)
                _render_artifact_paths(previous.values, output)
                return 0
            from medbenchagent.runtime.dependencies import verify_resume
            verify_resume(previous.values["context"])
            pending = [i for task in previous.tasks for i in task.interrupts]
            if pending:
                _render_interrupt(pending[0].value, output)
                next_input = Command(resume=_read_nonempty("Your response: ", input_fn, output))
            else:
                next_input = None

        while True:
            if experiment:
                update_experiment(settings.paths.runs_root, experiment["name"], status="running")
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
                    _render_update(payload, output)
                    interrupt = interrupt or _interrupt_payload(payload)

            snapshot = graph.get_state(config)
            known_run_directory = snapshot.values.get("run_directory") or known_run_directory
            if experiment:
                update_experiment(settings.paths.runs_root, experiment["name"], **snapshot_updates(snapshot))
            current_run_directory = snapshot.values.get("run_directory")
            if (
                isinstance(current_run_directory, str)
                and current_run_directory != announced_run_directory
            ):
                announced_run_directory = _announce_run(snapshot.values, output)
            config["recursion_limit"] = _recursion_limit(snapshot.values)
            if interrupt is not None:
                _render_interrupt(interrupt, output)
                answer = _read_nonempty("Your response: ", input_fn, output)
                next_input = Command(resume=answer)
                continue
            if snapshot.next:
                next_input = None
                continue
            _print_result(snapshot.values, output)
            return 0 if snapshot.values.get("status", "complete") == "complete" else 1
    except (EOFError, KeyboardInterrupt):
        interrupted = True
        output("")
        command = f"prepare --dataset-id {dataset_id}" if dataset_id else "build"
        recovery = f"experiment {experiment['name']}" if experiment else f"{command} --resume <resume identifier>"
        output(f"Run interrupted. Resume with: medbenchagent {recovery}.")
        if known_run_directory:
            output(f"Preserved run artifacts: {known_run_directory}")
        return 130
    except Exception as exc:
        failed = True
        output("")
        output(f"Benchmark construction failed: {type(exc).__name__}: {exc}")
        command = f"prepare --dataset-id {dataset_id}" if dataset_id else "build"
        recovery = f"experiment {experiment['name']}" if experiment else f"{command} --resume <resume identifier>"
        output(f"Files and checkpoints are preserved. Use {recovery}.")
        if known_run_directory:
            output(f"Preserved run artifacts: {known_run_directory}")
        return 1
    finally:
        if experiment and settings is not None:
            updates = {"status": "interrupted"} if interrupted or failed else {}
            if graph is not None and config:
                try:
                    updates = snapshot_updates(graph.get_state(config), interrupted=interrupted or failed)
                except Exception as exc:
                    # Diagnostics at the CLI boundary must preserve the original exit status.
                    output(f"Could not read checkpoint for experiment status: {type(exc).__name__}: {exc}")
            if known_run_directory:
                updates["run_directory"] = known_run_directory
            try:
                update_experiment(settings.paths.runs_root, experiment["name"], **updates)
            except Exception as exc:
                output(f"Could not update experiment registry: {type(exc).__name__}: {exc}")
        if checkpoint_connection is not None:
            checkpoint_connection.close()


def run_experiments(*, action=None, name=None, dataset_id=None, dataset_ids=None,
                    input_fn=input, output=print, settings=None):
    """A small terminal selector over named persistent threads."""
    from medbenchagent.settings import get_settings
    from medbenchagent.runtime.experiments import create_experiment, get_experiment, list_experiments
    settings = settings or get_settings()
    root = settings.paths.runs_root
    if action != "new" and (name or dataset_id or dataset_ids):
        raise ValueError("Creation options require 'experiment new <name>'")
    try:
        if action is None:
            entries = list_experiments(root)
            output("Experiments")
            for index, entry in enumerate(entries, 1):
                datasets = ", ".join(entry["datasets"]) or entry.get("dataset_id") or "—"
                output(f"{index}. {entry['name']} | {entry['status']} | {entry['mode']} | {datasets} | {entry['updated_at']}")
            output("[N] New experiment  [P] Prepare dataset  [Q] Quit")
            choice = _read_nonempty("Select number or name: ", input_fn, output)
            if choice.upper() == "Q":
                return 0
            if choice.upper() in {"N", "P"}:
                name = _read_nonempty("Experiment name: ", input_fn, output)
                if choice.upper() == "P":
                    dataset_id = _read_nonempty("Dataset ID: ", input_fn, output)
                action = "new"
            elif choice.isdigit() and 1 <= int(choice) <= len(entries):
                action = entries[int(choice) - 1]["name"]
            else:
                action = choice
        if action == "new":
            if not name:
                raise ValueError("Use experiment new <name>")
            entry = create_experiment(root, name, dataset_id=dataset_id, dataset_ids=dataset_ids)
        else:
            entry = get_experiment(root, action)
        return run_interactive_build(settings=settings, experiment=entry, input_fn=input_fn, output=output)
    except (EOFError, KeyboardInterrupt):
        output("Experiment selection cancelled.")
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medbenchagent")
    subcommands = parser.add_subparsers(dest="command")
    build = subcommands.add_parser(
        "build",
        help="Run the benchmark graph as an offline interactive terminal workflow.",
    )
    build.add_argument("--resume", help="Resume a persisted task by its printed identifier.")
    build.add_argument("--dataset", action="append", dest="dataset_ids", help="Select this dataset ID directly; repeat for multiple datasets.")
    experiment = subcommands.add_parser("experiment", help="Create, select or resume a named experiment.")
    experiment.add_argument("action", nargs="?", help="Existing experiment name, or new.")
    experiment.add_argument("name", nargs="?", help="Name for a new experiment.")
    experiment.add_argument("--prepare-dataset")
    experiment.add_argument("--dataset", action="append", dest="dataset_ids")
    prepare = subcommands.add_parser("prepare", help="Prepare and register one dataset from a natural-language source description.")
    prepare.add_argument("--dataset-id", required=True)
    prepare.add_argument("--resume")
    register = subcommands.add_parser("register", help="Register an already prepared bundle and Card.")
    register.add_argument("--dataset-id", required=True)
    register.add_argument("--bundle", required=True)
    refresh = subcommands.add_parser("refresh-card", help="Refresh dataset metadata without re-preparing data.")
    refresh.add_argument("--dataset-id", required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command in {None, "experiment"}:
        try:
            raise SystemExit(run_experiments(
                action=getattr(args, "action", None), name=getattr(args, "name", None),
                dataset_id=getattr(args, "prepare_dataset", None), dataset_ids=getattr(args, "dataset_ids", None)))
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "refresh-card":
        from medbenchagent.data.dataset_catalog import refresh_card
        _print_json(refresh_card(args.dataset_id), print)
        return
    if args.command == "register":
        from medbenchagent.data.dataset_catalog import register_dataset
        _print_json(register_dataset(args.dataset_id, args.bundle), print)
        return
    if args.command == "prepare":
        from medbenchagent.data.dataset_catalog import validate_dataset_id
        validate_dataset_id(args.dataset_id)
        raise SystemExit(run_interactive_build(resume=args.resume, dataset_id=args.dataset_id))
    if args.command == "build":
        exit_code = run_interactive_build(resume=args.resume, dataset_ids=args.dataset_ids)
        if exit_code:
            raise SystemExit(exit_code)
        return
    parser.print_help()
