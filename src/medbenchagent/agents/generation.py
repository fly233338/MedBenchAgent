"""GenerationAgent wrapper and private GenerationRequest adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from medbenchagent.agents.contracts import (
    GENERATION_AGENT,
    GenerationAgentInputState,
    GenerationAgentOutputState,
    GenerationAgentState,
)
from medbenchagent.contracts import GenerationRequest
from medbenchagent.construction import event_context
from medbenchagent.generation.templates import build_templates
from medbenchagent.harness import result
from medbenchagent.run_artifacts import atomic_write_json
from medbenchagent.generation.graph import compile_generation_subgraph
from medbenchagent.settings import get_settings


def _generation_request_node(state: GenerationAgentState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint contains no tasks for generation.")
    benchmark_info = blueprint.get("benchmark_info", state["benchmark_info"])
    defaults = get_settings().generation
    request: GenerationRequest = {
        "benchmark_info": benchmark_info,
        "run_id": state["run_id"],
        "dataset_id": state["selected_dataset_id"],
        "construction_events_path": state["construction_events_path"],
        "dataset_access_manifest": state["dataset_access_manifest"],
        "tasks": tasks,
        "target_sample_count": int(benchmark_info["sample_count"]),
        "seed": int(state.get("seed", defaults.seed)),
        "max_tasks_per_image": int(state.get("max_tasks_per_image", defaults.max_tasks_per_image)),
        "question_temperature": float(
            state.get("question_temperature", defaults.question_temperature)
        ),
        "output_directory": str(Path(state["working_blueprint_path"]).parent),
    }
    return {"generation_request": request}


def compile_generation_agent(*, checkpointer: Any = None):
    """Build the role-level generation adapter and unchanged generation loop."""
    graph = StateGraph(
        GenerationAgentState,
        input_schema=GenerationAgentInputState,
        output_schema=GenerationAgentOutputState,
    )
    graph.add_node("generation_request", _generation_request_node)
    graph.add_node("generate_raw_benchmark", compile_generation_subgraph())
    graph.add_edge(START, "generation_request")
    graph.add_edge("generation_request", "generate_raw_benchmark")
    graph.add_edge("generate_raw_benchmark", END)
    return graph.compile(checkpointer=checkpointer, name=GENERATION_AGENT)


def run_generation(context, arguments, graph):
    """Build/review templates and instantiate questions: {feedback?: string}. Requires a complete plan."""
    if not context.get("planned"):
        raise ValueError("Finish planning first")
    batch = Path(context["operation_directory"]) / "batch"
    batch.mkdir(parents=True, exist_ok=True)
    blueprint_file = batch / "blueprint.json"
    if not blueprint_file.is_file():
        blueprint = json.loads(Path(context["working_blueprint_path"]).read_text(encoding="utf-8"))
        for index, (task_id, task) in enumerate(blueprint["tasks"].items()):
            if task.get("sample_count", 0) == 0:
                continue
            saved = batch / ".templates" / f"{index}.json"
            if saved.is_file():
                templates = json.loads(saved.read_text(encoding="utf-8"))
            else:
                with event_context(agent="GenerationAgent", task_id=task_id, operation="templates"):
                    templates = build_templates(
                        task_id,
                        task,
                        context["benchmark_info"],
                        arguments.get("feedback", ""),
                        temperature=context.get("question_temperature"),
                    )
                atomic_write_json(saved, templates)
            task["question_templates"] = templates
        atomic_write_json(blueprint_file, blueprint)
        atomic_write_json(
            batch / "templates.json",
            {key: value.get("question_templates", []) for key, value in blueprint["tasks"].items()},
        )
    profile = json.loads(
        Path(context["dataset_access_manifest"]["profile_path"]).read_text(encoding="utf-8")
    )
    output = graph.invoke(
        dict(context, working_blueprint_path=str(blueprint_file)),
        config={"recursion_limit": max(1000, 4 * profile["total_images"] + 100)},
    )
    return result(
        "success",
        "Batch generated",
        **output,
        working_blueprint_path=str(blueprint_file),
        generated=True,
        audited=False,
    )
