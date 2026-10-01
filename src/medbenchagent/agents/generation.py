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
from medbenchagent.generation.contracts import GenerationRequest
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.generation.graph import compile_generation_subgraph
from medbenchagent.settings import get_settings
from medbenchagent.runtime.dependencies import publish, require
from medbenchagent.generation.v2 import generate_task_items, write_task_items


def _generation_request_node(state: GenerationAgentState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint contains no tasks for generation.")
    templates = json.loads((Path(state["working_blueprint_path"]).parent / "templates.json").read_text(encoding="utf-8"))
    tasks = {key: {**leaf, "question_templates": templates.get(key, [])} for key, leaf in tasks.items()}
    benchmark_info = blueprint.get("benchmark_info", state["benchmark_info"])
    defaults = get_settings().generation
    request: GenerationRequest = {
        "benchmark_info": benchmark_info,
        "run_id": state["run_id"],
        "dataset_id": state["selected_dataset_id"],
        "construction_events_path": state["construction_events_path"],
        "dataset_access_manifest": state["dataset_access_manifest"],
        "tasks": tasks,
        "target_sample_count": int(blueprint["allocation"]["allocated_count"]),
        "seed": int(state.get("seed", defaults.seed)),
        "max_tasks_per_image": int(state.get("max_tasks_per_image", defaults.max_tasks_per_image)),
        "max_total_tasks_per_image": state.get("max_total_tasks_per_image", defaults.max_total_tasks_per_image),
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


def run_generation(context, graph):
    """Instantiate questions deterministically from the approved plan, templates and allocation."""
    require(context, "Plan", "Templates", "Allocation")
    batch = Path(context["operation_directory"]) / "batch"
    batch.mkdir(parents=True, exist_ok=True)
    blueprint_file = batch / "blueprint.json"
    if not blueprint_file.is_file():
        blueprint = json.loads(Path(context["working_blueprint_path"]).read_text(encoding="utf-8"))
        approved = json.loads(Path(context["templates_path"]).read_text(encoding="utf-8"))
        atomic_write_json(batch / "templates.json", approved["templates"])
        atomic_write_json(blueprint_file, blueprint)
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
        **publish(
            context,
            "Generation",
            {"raw": output["raw_output_path"], "blueprint": str(blueprint_file)},
        ),
    )


def run_v2_generation(context, *, leaf_path=None, output_path=None, page_size=50):
    """Generate annotation.v5 benchmark items from compiled Task Leaves without legacy Protocols."""
    require(context, "Dataset")
    access = context["dataset_access_manifest"]
    if access.get("schema_version") != "annotation.v5":
        raise ValueError("v2 generation requires annotation.v5")
    leaf_path = leaf_path or context.get("task_leaves_path")
    if not leaf_path:
        raise ValueError("Compiled Task Leaves are required")
    leaves = json.loads(Path(leaf_path).read_text(encoding="utf-8"))
    from medbenchagent.data.v5_index import V5DatasetQuery
    query = V5DatasetQuery(access)
    cases = []
    offset = 0
    while True:
        page = query.sample(offset=offset, limit=page_size)
        cases.extend(page["values"])
        if page["complete"]:
            break
        offset = page["next_offset"]
    items = generate_task_items(cases, leaves.values(), context.get("selected_dataset_id", "dataset"))
    destination = output_path or str(Path(context["operation_directory"]) / "benchmark_items.v2.jsonl")
    write_task_items(destination, items)
    return result("success", f"Generated {len(items)} v2 benchmark items", raw_output_path=destination,
                  item_count=len(items), task_counts={leaf_id: sum(item["task_id"] == leaf_id for item in items) for leaf_id in leaves})
