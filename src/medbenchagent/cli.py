"""Command-line entry points for local benchmark construction and finalization."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable

from medbenchagent.answer_balance import balance_jsonl

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]

_QUESTION_TEXT = {
    "benchmark_info_confirmation": "Confirm the BenchmarkInfo or describe the changes you need.",
    "dataset_selection_confirmation": "Confirm the recommended dataset or name another available dataset.",
    "task_plan_review": "Confirm the Task Plan or describe one task that should be revised.",
    "sample_count_confirmation": "Confirm the per-task sample counts or describe adjustments.",
    "materials_choose": "Choose one or more medical PDFs by filename or describe the references you need.",
    "question_design_confirmation": "Confirm the question designs or describe the tasks to revise.",
}

_NODE_LABELS = {
    "run_initialize": "initialize run directory",
    "SpecificationAgent": "parse benchmark requirement",
    "CapabilityAgent": "plan dataset and tasks",
    "ClinicalDesignAgent": "design medical questions",
    "rag_design_export": "export RAG designs",
    "GenerationAgent": "generate benchmark",
    "generation_summary": "summarize generation",
    "AuditAgent": "audit benchmark",
    "run_finalize": "finalize run artifacts",
}

_ARTIFACT_PATH_KEYS = {
    "working_blueprint_path",
    "effective_config_path",
    "construction_events_path",
    "task_screening_report_path",
    "rag_designs_path",
    "raw_output_path",
    "provenance_output_path",
    "construction_report_path",
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
    from medbenchagent.dataset_catalog import load_dataset_catalog, resolve_annotation_path
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

    try:
        datasets = load_dataset_catalog(settings.paths.dataset_catalog)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        viable_datasets = 0
        dataset_problems: list[str] = []
        for dataset in datasets:
            if not dataset["available"]:
                continue
            annotation_path = resolve_annotation_path(dataset)
            missing: list[str] = []
            if not annotation_path.is_file():
                missing.append(f"annotation.jsonl: {annotation_path}")
            image_manifest = annotation_path.parent / "image_manifest.jsonl"
            if not image_manifest.is_file():
                missing.append(f"image_manifest.jsonl: {image_manifest}")
            if missing:
                dataset_problems.append(f"{dataset['id']} is missing " + "; ".join(missing))
            else:
                viable_datasets += 1
        if viable_datasets == 0:
            errors.append("No dataset is ready for a complete run: " + "; ".join(dataset_problems))
    return errors


def _print_json(value: object, output: OutputFunction) -> None:
    output(json.dumps(value, ensure_ascii=False, indent=2))


def _render_interrupt(payload: dict[str, Any], output: OutputFunction) -> None:
    kind = str(payload.get("kind", ""))
    output("")
    output(f"=== {_QUESTION_TEXT.get(kind, 'Your confirmation is required.')} ===")
    if kind == "benchmark_info_confirmation":
        _print_json(payload.get("benchmark_info", {}), output)
    elif kind == "dataset_selection_confirmation":
        output("Recommended dataset:")
        _print_json(payload.get("recommendation", {}), output)
        datasets = payload.get("datasets", [])
        output("Available datasets: " + ", ".join(
            f"{item.get('id')} ({item.get('name')})"
            for item in datasets
            if isinstance(item, dict) and item.get("available")
        ))
    elif kind == "task_plan_review":
        profile = payload.get("dataset_profile", {})
        if isinstance(profile, dict):
            output(
                "Dataset profile: "
                f"{profile.get('total_images', '?')} images, "
                f"{len(profile.get('fields', {}))} fields"
            )
        tasks = payload.get("task_plan", {}).get("tasks", {})
        for task_id, decision in tasks.items():
            fields = decision.get("supporting_fields", [])
            if isinstance(decision.get("question_types"), dict):
                fields = {
                    question_type_id: question_type.get("supporting_fields", [])
                    for question_type_id, question_type in decision["question_types"].items()
                    if isinstance(question_type, dict)
                }
            output(
                f"- {task_id}: {decision.get('decision')} | "
                f"fields={fields} | "
                f"{decision.get('reason', '')}"
            )
    elif kind == "sample_count_confirmation":
        output(f"Total questions: {payload.get('total')}")
        _print_json(payload.get("sample_counts", {}), output)
    elif kind == "materials_choose":
        if payload.get("error"):
            output(f"Previous selection was invalid: {payload['error']}")
        for material in payload.get("materials", []):
            if isinstance(material, dict):
                output(f"- [{material.get('category_label')}] {material.get('file')}")
    elif kind == "question_design_confirmation":
        for task_id, task in payload.get("tasks", {}).items():
            design = {
                key: task[key]
                for key in (
                    "question_templates",
                    "options",
                    "answer_field",
                    "answer_value_mapping",
                    "generation_logic",
                    "medical_task_description",
                    "citations",
                )
                if key in task
            }
            output(f"{task_id}:")
            _print_json(design, output)
    else:
        _print_json(payload, output)
    output(_QUESTION_TEXT.get(kind, str(payload.get("question", "Enter your response."))))


def _interrupt_payload(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        interrupts = value.get("__interrupt__")
        if isinstance(interrupts, (tuple, list)) and interrupts:
            payload = getattr(interrupts[0], "value", None)
            if isinstance(payload, dict):
                return payload
        for nested in value.values():
            if payload := _interrupt_payload(nested):
                return payload
    elif isinstance(value, (tuple, list)):
        for nested in value:
            if payload := _interrupt_payload(nested):
                return payload
    return None


def _event_parts(event: object) -> tuple[tuple[str, ...], str, object]:
    if isinstance(event, tuple) and len(event) == 3:
        namespace, mode, payload = event
        if isinstance(namespace, tuple) and isinstance(mode, str):
            return namespace, mode, payload
    return (), "updates", event


def _node_name(namespace: tuple[str, ...], payload: object) -> str:
    names = [part.split(":", 1)[0] for part in namespace]
    if isinstance(payload, dict):
        names.extend(key for key in payload if key != "__interrupt__")
    return " / ".join(names)


def _render_artifact_paths(payload: object, output: OutputFunction) -> None:
    found: list[tuple[str, str]] = []

    def collect(value: object, *, in_audit_artifacts: bool = False) -> None:
        if not isinstance(value, dict):
            return
        for key, nested in value.items():
            is_audit_artifacts = key == "audit_artifacts"
            if isinstance(nested, str) and (
                key in _ARTIFACT_PATH_KEYS
                or (in_audit_artifacts and key in _AUDIT_ARTIFACT_KEYS)
            ):
                found.append((key, nested))
            elif isinstance(nested, dict):
                collect(nested, in_audit_artifacts=is_audit_artifacts)

    collect(payload)
    for key, path in dict.fromkeys(found):
        output(f"[artifact] {key}: {path}")


def _render_update(namespace: tuple[str, ...], payload: object, output: OutputFunction) -> None:
    _render_artifact_paths(payload, output)
    if interrupt := _interrupt_payload(payload):
        kind = str(interrupt.get("kind", ""))
        output(f"[waiting] {_QUESTION_TEXT.get(kind, 'user confirmation')}")
        return
    name = _node_name(namespace, payload)
    if not name or any(part in name for part in ("generation_step", "rag_design_task")):
        return
    leaf = name.rsplit(" / ", 1)[-1]
    output(f"[done] {_NODE_LABELS.get(leaf, name)}")


def _render_task(namespace: tuple[str, ...], payload: object, output: OutputFunction) -> None:
    if not isinstance(payload, dict) or "input" not in payload:
        return
    task_name = payload.get("name")
    if not isinstance(task_name, str):
        return
    names = [part.split(":", 1)[0] for part in namespace] + [task_name]
    name = " / ".join(names)
    if any(part in name for part in ("generation_step", "rag_design_task")):
        return
    output(f"[start] {_NODE_LABELS.get(task_name, name)}")


def _render_progress(payload: object, output: OutputFunction) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("kind") == "rag_progress":
        output(
            f"[RAG] {payload.get('task_id')}: "
            f"{payload.get('completed')}/{payload.get('total')}, "
            f"remaining={payload.get('remaining')}"
        )
    elif payload.get("kind") == "generation_progress":
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
    output("=== Benchmark construction completed ===")
    output(f"Run ID: {values.get('run_id')}")
    run_directory = values.get("run_directory")
    if not run_directory and isinstance(values.get("manifest_path"), str):
        run_directory = str(Path(values["manifest_path"]).parent)
    output(f"Run directory: {run_directory}")
    output(f"Total items: {values.get('item_count', 0)}")
    output(f"Task counts: {values.get('task_counts', {})}")
    output(f"Shortfalls: {values.get('generation_shortfalls', {})}")
    output(f"Raw JSONL: {values.get('raw_output_path')}")
    artifacts = values.get("audit_artifacts", {})
    if isinstance(artifacts, dict):
        output(f"Audited JSONL: {artifacts.get('audited')}")
        output(f"Flagged JSONL: {artifacts.get('flagged')}")
        output(f"Rejected JSONL: {artifacts.get('rejected')}")
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


def _run_directory(graph: object, config: dict[str, Any] | None) -> str | None:
    if graph is None or config is None:
        return None
    try:
        snapshot = graph.get_state(config)
    except Exception:
        return None
    values = getattr(snapshot, "values", None)
    path = values.get("run_directory") if isinstance(values, dict) else None
    return path if isinstance(path, str) and path else None


def run_interactive_build(
    *,
    input_fn: InputFunction = input,
    output: OutputFunction = print,
    settings: Any = None,
    graph: Any = None,
) -> int:
    """Run the benchmark graph as a local, terminal-driven conversation."""
    _enforce_offline_mode()
    config: dict[str, Any] | None = None
    announced_run_directory: str | None = None
    try:
        requirement = _read_nonempty(
            "Describe the Medical VLM benchmark to build: ", input_fn, output
        )
        if settings is None:
            from medbenchagent.settings import get_settings

            settings = get_settings()
        _print_settings(settings, output)
        errors = _preflight_errors(settings)
        if errors:
            output("")
            output("Local preflight failed:")
            for error in errors:
                output(f"- {error}")
            return 2

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        if graph is None:
            from medbenchagent.workflow.graph import compile_benchmark_graph

            graph = compile_benchmark_graph(checkpointer=MemorySaver())
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 1_000,
        }
        next_input: object = {"user_requirement": requirement}

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
                    _render_update(namespace, payload, output)
                    interrupt = interrupt or _interrupt_payload(payload)

            snapshot = graph.get_state(config)
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
            return 0
    except (EOFError, KeyboardInterrupt):
        output("")
        output(
            "Run interrupted. This CLI uses an in-memory checkpoint and cannot resume "
            "after the process exits."
        )
        if run_directory := _run_directory(graph, config):
            output(f"Preserved run artifacts: {run_directory}")
        return 130
    except Exception as exc:
        output("")
        output(f"Benchmark construction failed: {type(exc).__name__}: {exc}")
        output(
            "Files already written are preserved, but this CLI cannot resume the Graph "
            "after the process exits."
        )
        if run_directory := _run_directory(graph, config):
            output(f"Preserved run artifacts: {run_directory}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medbenchagent")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser(
        "build",
        help="Run the benchmark graph as an offline interactive terminal workflow.",
    )
    balance = subcommands.add_parser("balance-answers", help="Write a balanced JSONL copy.")
    balance.add_argument("--input", required=True, dest="input_path")
    balance.add_argument("--output", required=True, dest="output_path")
    balance.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "build":
        exit_code = run_interactive_build()
        if exit_code:
            raise SystemExit(exit_code)
        return
    if args.command == "balance-answers":
        print(balance_jsonl(args.input_path, args.output_path, args.seed))
        return
    parser.print_help()
