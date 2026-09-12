"""Supervisor-driven benchmark construction with persistent tool boundaries."""

from __future__ import annotations
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict
from langchain_core.messages import convert_to_messages
from langgraph.graph import StateGraph, START, END
from medbenchagent.agents.generation import compile_generation_agent, run_generation
from medbenchagent.agents.audit import run_audit
from medbenchagent.agents.task_planning import compile_task_planning_agent
from medbenchagent.planning.benchmark_info import (
    extract_benchmark_info,
    missing_fields,
    write_working_blueprint,
)
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.tool_contracts import FeedbackArguments, PrepareDataArguments
from medbenchagent.runtime.run_artifacts import initialize_run
from medbenchagent.runtime.run_export import finalize_run
from medbenchagent.settings import get_settings, run_settings
from medbenchagent.runtime.dependencies import publish, require, initialize_dependencies, verify_resume


class BenchmarkInputState(TypedDict, total=False):
    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    question_temperature: float


class BenchmarkState(BenchmarkInputState, total=False):
    context: dict[str, Any]
    result: dict[str, Any]
    run_id: str
    run_directory: str
    working_blueprint_path: str
    construction_events_path: str
    benchmark_info: dict
    selected_dataset_id: str
    raw_output_path: str
    audit_artifacts: dict
    manifest_path: str
    benchmark_id: str
    generation_shortfalls: dict
    item_count: int
    task_counts: dict
    status: str
    summary: str


class BenchmarkOutputState(TypedDict, total=False):
    run_id: str
    run_directory: str
    working_blueprint_path: str
    construction_events_path: str
    benchmark_info: dict
    selected_dataset_id: str
    raw_output_path: str
    audit_artifacts: dict
    manifest_path: str
    benchmark_id: str
    generation_shortfalls: dict
    item_count: int
    task_counts: dict
    status: str
    summary: str


def specify(context):
    """Extract or correct the benchmark brief from the user's requirement."""
    info = extract_benchmark_info(context["user_requirement"], context.get("benchmark_info"))
    if missing := missing_fields(info):
        return result(
            "success",
            "Brief needs clarification: " + ", ".join(missing),
            benchmark_info=info,
            brief_ready=False,
        )
    path = Path(context["operation_directory"]) / "blueprint.json"
    write_working_blueprint(info, path)
    return result(
        "success",
        "Benchmark brief extracted",
        benchmark_info=info,
        brief_ready=True,
        working_blueprint_path=str(path),
        **publish(context, "Requirement", {"path": str(path), "brief": info}, ()),
    )


def _child_updates(output):
    return {
        k: v
        for k, v in output["context"].items()
        if k not in {"operation_directory", "feedback", "dataset_profile"}
    }


def describe(context):
    return {
        key: context.get(key)
        for key in (
            "user_requirement",
            "benchmark_info",
            "brief_ready",
            "selected_dataset_id",
            "annotation_path",
            "artifact_versions",
            "global_issue",
            "generation_shortfalls",
            "audit_summary",
            "last_agent_result",
            "reworks",
        )
    }


def compile_benchmark_graph(*, checkpointer=None):
    planning = compile_task_planning_agent()
    generation = compile_generation_agent()

    def plan(context, *, feedback: str = ""):
        """Delegate task planning. Requires a complete benchmark brief."""
        if not context.get("brief_ready"):
            raise ValueError("Extract and clarify the brief first")
        child_context = dict(context, feedback=feedback)
        output = planning.invoke({"context": child_context})
        updates = _child_updates(output)
        updates.update(last_agent_result=output["result"]["summary"])
        return result(
            "success" if output["result"]["status"] in {"complete", "escalate"} else "failed", output["result"]["summary"], **updates
        )

    def prepare_data(context, *, source_root: str, dataset_id: str, feedback: str = ""):
        """Prepare raw 2D data. Standard data skips this."""
        from medbenchagent.agents.data_processing import (
            compile_data_processing_agent,
            preparation_context,
        )

        source = source_root
        if context.get("original_source_root") != str(Path(source).resolve()) and not Path(source).is_dir():
            raise ValueError("Provide the existing raw source directory from the user's request")
        prepared = preparation_context(
            context, str(Path(source).resolve()), dataset_id
        )
        prepared["feedback"] = feedback
        output = compile_data_processing_agent().invoke({"context": prepared})
        updates = _child_updates(output)
        updates.update(
            last_agent_result=output["result"]["summary"],
        )
        return result(
            "success" if output["result"]["status"] in {"complete", "escalate"} else "failed",
            output["result"]["summary"],
            **updates,
        )

    def generate(context, *, feedback: str = ""):
        """Build/review templates and instantiate questions. Requires a complete plan."""
        return run_generation(context, generation, feedback=feedback)

    def finish(context):
        require(context, "Audit")
        directory = Path(context["run_directory"])
        published = {}
        for key, source in {
            "working_blueprint_path": context["working_blueprint_path"],
            "raw_output_path": context["raw_output_path"],
            **context["audit_artifacts"],
        }.items():
            source = Path(source)
            destination = directory / source.name
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
            published[key] = str(destination)
        templates = Path(context["working_blueprint_path"]).parent / "templates.json"
        if templates.is_file():
            shutil.copyfile(templates, directory / "templates.json")
        context.update(
            working_blueprint_path=published["working_blueprint_path"],
            raw_output_path=published["raw_output_path"],
            audit_artifacts={"audited": published["audited"], "report": published["report"]},
        )
        blueprint = json.loads(Path(context["working_blueprint_path"]).read_text(encoding="utf-8"))
        counts = Counter(
            json.loads(line)["task_id"]
            for line in Path(published["audited"]).read_text(encoding="utf-8").splitlines()
            if line
        )
        shortfalls = {
            key: max(0, task["sample_count"] - counts[key])
            for key, task in blueprint["tasks"].items()
        }
        identifier, manifest = finalize_run(
            run_directory=context["run_directory"],
            run_id=context["run_id"],
            blueprint_path=context["working_blueprint_path"],
            raw_output_path=context["raw_output_path"],
            construction_events_path=context["construction_events_path"],
            dataset_catalog_entry=context["dataset_entry"],
            dataset_profile=json.loads(
                Path(context["dataset_access_manifest"]["profile_path"]).read_text(encoding="utf-8")
            ),
            dataset_access_manifest=context["dataset_access_manifest"],
            image_manifest_path=Path(context["annotation_path"]).parent / "image_manifest.jsonl",
            audit_artifacts=context["audit_artifacts"],
        )
        return result(
            "complete",
            "Benchmark completed; any deficits are reported",
            manifest_path=manifest,
            benchmark_id=identifier,
            working_blueprint_path=context["working_blueprint_path"],
            raw_output_path=context["raw_output_path"],
            audit_artifacts=context["audit_artifacts"],
            item_count=sum(counts.values()),
            task_counts=dict(counts),
            generation_shortfalls=shortfalls,
        )

    supervisor = compile_agent(
        "SupervisorAgent",
        "Build the requested Medical VLM benchmark. Extract the brief. If the user supplies raw data, prepare it; "
        "otherwise planning can select catalog data. Delegate planning, generation and audit as needed. "
        "Ask the user about global goals or a global_issue escalated by a specialist. Specialists ask local questions directly. "
        "No routine stage confirmations. Review audit feedback and shortfalls before finishing; rework the "
        "responsible module only when evidence suggests it can help. Never silently lower the requested target. "
        "Finite data exhaustion can finish with explicit shortfalls. Semantic flags are advisory, not automatic rejection.",
        {
            "specify": specify,
            "plan": plan,
            "prepare_data": prepare_data,
            "generate": generate,
            "audit": run_audit,
        },
        describe,
        finish,
        args_schemas={
            "plan": FeedbackArguments,
            "prepare_data": PrepareDataArguments,
            "generate": FeedbackArguments,
        },
    )

    def initialize(state):
        overrides = {
            key: state[key]
            for key in ("seed", "max_tasks_per_image", "question_temperature")
            if key in state
        }
        requirement = state.get("user_requirement") or next(
            (
                getattr(m, "content", "")
                for m in reversed(convert_to_messages(state.get("messages", [])))
                if getattr(m, "type", "") == "human"
            ),
            "",
        )
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("A natural-language benchmark requirement is required")
        settings = get_settings()
        defaults = {
            key: getattr(settings.generation, key)
            for key in ("seed", "max_tasks_per_image", "question_temperature")
        }
        run = initialize_run(graph_overrides=overrides)
        context = dict(
            run,
            **(defaults | overrides),
            user_requirement=requirement,
            operation_directory=str(Path(run["run_directory"]) / ".runtime"),
            limits=settings.harness,
        )
        initialize_dependencies(context, settings)
        return dict(run, context=context)

    def run_supervisor(state):
        verify_resume(state["context"])
        with run_settings(state["context"]["effective_config_path"]):
            output = supervisor.invoke({"context": state["context"]})
        public = {
            key: value
            for key, value in output["context"].items()
            if key in BenchmarkState.__annotations__
        }
        return dict(
            public,
            context=output["context"],
            result=output["result"],
            status=output["result"]["status"],
            summary=output["result"]["summary"],
        )

    graph = StateGraph(
        BenchmarkState, input_schema=BenchmarkInputState, output_schema=BenchmarkOutputState
    )
    graph.add_node("initialize", initialize)
    graph.add_node("supervisor", run_supervisor)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile(checkpointer=checkpointer)


benchmark_graph = compile_benchmark_graph()
