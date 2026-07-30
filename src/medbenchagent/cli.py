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
    "benchmark_info_confirmation": "请确认 BenchmarkInfo，或描述需要修改的内容。",
    "dataset_selection_confirmation": "请确认推荐的数据集，或输入另一个可用数据集。",
    "task_plan_review": "请确认 Task Plan，或说明要修改的一个任务及修改要求。",
    "sample_count_confirmation": "请确认各任务题量，或描述需要调整的数量。",
    "materials_choose": "请选择一个或多个医学 PDF，可直接输入文件名或描述所需资料。",
    "question_design_confirmation": "请确认题目设计，或描述需要修改的任务和要求。",
}

_NODE_LABELS = {
    "run_initialize": "初始化运行目录",
    "SpecificationAgent": "需求解析",
    "CapabilityAgent": "数据与任务规划",
    "ClinicalDesignAgent": "医学题目设计",
    "rag_design_export": "导出 RAG 设计",
    "GenerationAgent": "生成 benchmark",
    "generation_summary": "汇总生成结果",
    "AuditAgent": "审计 benchmark",
    "run_finalize": "完成运行产物",
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
        output("输入不能为空，请重新输入。")


def _preflight_errors(settings: Any) -> list[str]:
    """Return all local-resource problems that would prevent a complete run."""
    from medbenchagent.dataset_catalog import load_dataset_catalog, resolve_annotation_path
    from medbenchagent.rag.pdf import RagUnavailable, discover_medical_materials

    errors: list[str] = []
    if not settings.paths.blueprint_template.is_file():
        errors.append(f"Blueprint 模板不存在：{settings.paths.blueprint_template}")

    for label, path in (
        ("embedding 模型", settings.rag.embedding_model),
        ("reranker 模型", settings.rag.reranker_model),
    ):
        if path is None:
            errors.append(f"{label}尚未配置")
        elif not path.is_dir():
            errors.append(f"{label}目录不存在：{path}")

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
                missing.append(f"annotation.jsonl：{annotation_path}")
            image_manifest = annotation_path.parent / "image_manifest.jsonl"
            if not image_manifest.is_file():
                missing.append(f"image_manifest.jsonl：{image_manifest}")
            if missing:
                dataset_problems.append(f"{dataset['id']} 缺少 " + "；".join(missing))
            else:
                viable_datasets += 1
        if viable_datasets == 0:
            errors.append("没有可完整运行的数据集：" + "；".join(dataset_problems))
    return errors


def _print_json(value: object, output: OutputFunction) -> None:
    output(json.dumps(value, ensure_ascii=False, indent=2))


def _render_interrupt(payload: dict[str, Any], output: OutputFunction) -> None:
    kind = str(payload.get("kind", ""))
    output("")
    output(f"=== {_QUESTION_TEXT.get(kind, '需要你的确认')} ===")
    if kind == "benchmark_info_confirmation":
        _print_json(payload.get("benchmark_info", {}), output)
    elif kind == "dataset_selection_confirmation":
        output("推荐数据集：")
        _print_json(payload.get("recommendation", {}), output)
        datasets = payload.get("datasets", [])
        output("可用数据集：" + "、".join(
            f"{item.get('id')} ({item.get('name')})"
            for item in datasets
            if isinstance(item, dict) and item.get("available")
        ))
    elif kind == "task_plan_review":
        profile = payload.get("dataset_profile", {})
        if isinstance(profile, dict):
            output(
                "数据集概况："
                f"{profile.get('total_images', '?')} 张图像，"
                f"{len(profile.get('fields', {}))} 个字段"
            )
        tasks = payload.get("task_plan", {}).get("tasks", {})
        for task_id, decision in tasks.items():
            output(
                f"- {task_id}: {decision.get('decision')} | "
                f"fields={decision.get('supporting_fields', [])} | "
                f"{decision.get('reason', '')}"
            )
    elif kind == "sample_count_confirmation":
        output(f"总题量：{payload.get('total')}")
        _print_json(payload.get("sample_counts", {}), output)
    elif kind == "materials_choose":
        if payload.get("error"):
            output(f"上次选择无效：{payload['error']}")
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
    output(_QUESTION_TEXT.get(kind, str(payload.get("question", "请输入你的回复。"))))


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


def _render_update(namespace: tuple[str, ...], payload: object, output: OutputFunction) -> None:
    name = _node_name(namespace, payload)
    if not name or any(part in name for part in ("generation_step", "rag_design_task")):
        return
    leaf = name.rsplit(" / ", 1)[-1]
    output(f"[完成] {_NODE_LABELS.get(leaf, name)}")


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
    output(f"[开始] {_NODE_LABELS.get(task_name, name)}")


def _render_progress(payload: object, output: OutputFunction) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("kind") == "rag_progress":
        output(
            f"[RAG] {payload.get('task_id')}："
            f"{payload.get('completed')}/{payload.get('total')}，"
            f"剩余 {payload.get('remaining')}"
        )
    elif payload.get("kind") == "generation_progress":
        output(
            f"[生成] stage={payload.get('stage') or 'complete'} "
            f"cursor={payload.get('cursor')} "
            f"完成={payload.get('completed_counts', {})}/"
            f"目标={payload.get('target_counts', {})} "
            f"拒绝={payload.get('rejections', {})} "
            f"耗尽={payload.get('exhausted')}"
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
    output("=== 生效配置 ===")
    output(f"LLM：{settings.llm.model} @ {settings.llm.base_url}")
    output(
        "生成："
        f"seed={settings.generation.seed}, "
        f"max_tasks_per_image={settings.generation.max_tasks_per_image}, "
        f"question_temperature={settings.generation.question_temperature}"
    )
    output(f"Embedding：{settings.rag.embedding_model}")
    output(f"Reranker：{settings.rag.reranker_model}")
    output(f"数据集目录：{settings.paths.dataset_catalog}")
    output(f"医学资料目录：{settings.paths.materials_root}")
    output(f"运行产物目录：{settings.paths.runs_root}")


def _print_result(values: dict[str, Any], output: OutputFunction) -> None:
    output("")
    output("=== Benchmark 构建完成 ===")
    output(f"Run ID：{values.get('run_id')}")
    output(f"运行目录：{values.get('run_directory')}")
    output(f"题目总数：{values.get('item_count', 0)}")
    output(f"各任务数量：{values.get('task_counts', {})}")
    output(f"Shortfall：{values.get('generation_shortfalls', {})}")
    output(f"Raw JSONL：{values.get('raw_output_path')}")
    artifacts = values.get("audit_artifacts", {})
    if isinstance(artifacts, dict):
        output(f"Audited JSONL：{artifacts.get('audited')}")
        output(f"Flagged JSONL：{artifacts.get('flagged')}")
        output(f"Rejected JSONL：{artifacts.get('rejected')}")
        output(f"Audit report：{artifacts.get('report')}")
        report_path = artifacts.get("report")
        if isinstance(report_path, str) and Path(report_path).is_file():
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            output(
                "审计摘要："
                f"通过={report.get('hard_audit', {}).get('accepted', 0)}，"
                f"拒绝={report.get('hard_audit', {}).get('rejected', 0)}，"
                f"语义标记={report.get('semantic_audit', {}).get('flagged', 0)}"
            )
    output(f"Manifest：{values.get('manifest_path')}")


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
    try:
        requirement = _read_nonempty(
            "请描述要构建的 Medical VLM benchmark：", input_fn, output
        )
        if settings is None:
            from medbenchagent.settings import get_settings

            settings = get_settings()
        _print_settings(settings, output)
        errors = _preflight_errors(settings)
        if errors:
            output("")
            output("本地运行预检未通过：")
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
            config["recursion_limit"] = _recursion_limit(snapshot.values)
            if interrupt is not None:
                _render_interrupt(interrupt, output)
                answer = _read_nonempty("你的回复：", input_fn, output)
                next_input = Command(resume=answer)
                continue
            if snapshot.next:
                next_input = None
                continue
            _print_result(snapshot.values, output)
            return 0
    except (EOFError, KeyboardInterrupt):
        output("")
        output("运行已中止。当前 CLI 使用进程内 checkpoint，退出后不能从中断点恢复。")
        if run_directory := _run_directory(graph, config):
            output(f"已写入的运行产物：{run_directory}")
        return 130
    except Exception as exc:
        output("")
        output(f"Benchmark 构建失败：{type(exc).__name__}: {exc}")
        output("已写入的运行产物会保留；当前 CLI 退出后不能从 Graph 中断点恢复。")
        if run_directory := _run_directory(graph, config):
            output(f"已写入的运行产物：{run_directory}")
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
