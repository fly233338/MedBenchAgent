"""LangGraph subgraph for RAG-backed question design and review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from medbenchagent.llm import complete_json
from medbenchagent.construction import event_context, record_event
from medbenchagent.rag.pdf import LocalPdfRag, design_task, write_task_design

PROJECT_ROOT = Path(__file__).parents[3]
MATERIALS_ROOT = PROJECT_ROOT / "materials"


class RagDesignInputState(TypedDict):
    """The fixed inputs accepted by the private RAG-design subgraph."""

    benchmark_info: dict[str, Any]
    working_blueprint_path: str
    selected_material_files: list[str]
    dataset_profile: dict[str, Any]
    run_id: str
    selected_dataset_id: str
    construction_events_path: str


class RagDesignState(RagDesignInputState, total=False):
    """Private, checkpointed state for the RAG design and review loop."""

    rag_task_queue: list[str]
    rag_task_feedback: dict[str, str]
    rag_current_task_id: str
    rag_completed_count: int
    rag_total_count: int
    designs_confirmed: bool


class RagDesignOutputState(TypedDict):
    """Stable result returned from the RAG-design subgraph to the root graph."""

    working_blueprint_path: str


def _emit_custom_event(event: dict[str, Any]) -> None:
    try:
        get_stream_writer()(event)
    except RuntimeError:
        return


def _rag_prepare_node(state: RagDesignState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint must retain at least one task before RAG design.")
    queue = [task_id for task_id, task in tasks.items() if task.get("sample_count", 0) > 0]
    if not queue:
        raise ValueError("Working Blueprint must allocate samples to at least one task.")
    return {
        "rag_task_queue": queue,
        "rag_task_feedback": {},
        "rag_current_task_id": "",
        "rag_completed_count": 0,
        "rag_total_count": len(queue),
        "designs_confirmed": False,
    }


def _rag_design_task_node(state: RagDesignState) -> dict[str, Any]:
    queue = state.get("rag_task_queue", [])
    if not queue:
        raise ValueError("RAG task queue is empty.")
    task_id = queue[0]
    blueprint_path = Path(state["working_blueprint_path"])
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    task = blueprint.get("tasks", {}).get(task_id)
    if not isinstance(task, dict):
        raise ValueError(f"Working Blueprint does not contain queued task: {task_id}")
    rag = LocalPdfRag(
        MATERIALS_ROOT,
        PROJECT_ROOT / "benchmarks" / "rag_cache",
        state["selected_material_files"],
    )
    event_values = {
        "construction_events_path": state.get("construction_events_path"),
        "run_id": state.get("run_id"),
        "dataset_id": state.get("selected_dataset_id"),
        "task_id": task_id,
        "agent": "rag",
        "operation": "rag_design",
    }
    with event_context(**event_values):
        design = design_task(
            rag,
            blueprint.get("benchmark_info", state["benchmark_info"]),
            task_id,
            task,
            dataset_profile=state["dataset_profile"],
            feedback=state.get("rag_task_feedback", {}).get(task_id, ""),
        )
    if state.get("construction_events_path"):
        record_event(
            state["construction_events_path"], "rag_retrieval", run_id=state.get("run_id"),
            dataset_id=state.get("selected_dataset_id"), task_id=task_id, agent="rag",
            operation="retrieve", status="success"
        )
    write_task_design(blueprint_path, task_id, design)
    remaining = queue[1:]
    completed = state.get("rag_completed_count", 0) + 1
    _emit_custom_event(
        {
            "kind": "rag_progress",
            "task_id": task_id,
            "completed": completed,
            "total": state.get("rag_total_count", len(queue)),
            "remaining": len(remaining),
        }
    )
    return {
        "rag_task_queue": remaining,
        "rag_current_task_id": task_id,
        "rag_completed_count": completed,
        "designs_confirmed": False,
    }


def _validate_design_review_action(response: object, task_ids: list[str]) -> dict[str, Any]:
    if response == {"action": "confirm"}:
        return {"action": "confirm"}
    if not isinstance(response, dict) or set(response) != {"action", "revisions"}:
        raise ValueError("Design review must confirm or provide revisions.")
    revisions = response.get("revisions")
    if response.get("action") != "revise" or not isinstance(revisions, list) or not revisions:
        raise ValueError("Design revisions must be a non-empty list.")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for revision in revisions:
        if not isinstance(revision, dict) or set(revision) != {"task_id", "feedback"}:
            raise ValueError("Each design revision must contain task_id and feedback.")
        task_id = revision.get("task_id")
        feedback = revision.get("feedback")
        if task_id not in task_ids or task_id in seen:
            raise ValueError("Design revisions must target unique current Blueprint tasks.")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("Design revision feedback must be non-empty.")
        seen.add(task_id)
        normalized.append({"task_id": task_id, "feedback": feedback.strip()})
    return {"action": "revise", "revisions": normalized}


def _design_review_action(answer: object, tasks: dict[str, Any]) -> dict[str, Any]:
    task_ids = list(tasks)
    prompt = (
        "Convert the user's review into exactly one of these JSON shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","revisions":[{"task_id":"exact task id","feedback":"requested design change"}]}\n'
        "Confirm only when the complete Blueprint task design is accepted. Support multiple task revisions "
        "and expand a global request to every affected task. Revisions may change only question_templates, "
        "options, generation_logic, or medical_task_description. They must not add or remove tasks or alter "
        "supporting_fields, task_type, or response_format.\n"
        f"Current tasks: {json.dumps(tasks, ensure_ascii=False)}\nUser response: {answer}"
    )
    response = complete_json("Interpret a Question Design review response. Return JSON only.", prompt)
    try:
        return _validate_design_review_action(response, task_ids)
    except ValueError as first_error:
        repaired = complete_json(
            "Repair a Question Design review action. Return JSON only.",
            f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
            f"Contract error: {first_error}",
        )
        return _validate_design_review_action(repaired, task_ids)


def _design_review_node(state: RagDesignState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = {
        task_id: task
        for task_id, task in blueprint.get("tasks", {}).items()
        if task.get("sample_count", 0) > 0
    }
    answer = interrupt(
        {
            "kind": "question_design_confirmation",
            "tasks": tasks,
            "question": "Confirm the Blueprint task designs or describe one or more corrections.",
        }
    )
    action = _design_review_action(answer, tasks)
    if action["action"] == "confirm":
        return {"designs_confirmed": True, "rag_task_queue": [], "rag_task_feedback": {}}
    feedback = {revision["task_id"]: revision["feedback"] for revision in action["revisions"]}
    return {
        "designs_confirmed": False,
        "rag_task_queue": list(feedback),
        "rag_task_feedback": feedback,
        "rag_current_task_id": "",
        "rag_completed_count": 0,
        "rag_total_count": len(feedback),
    }


def compile_rag_designs_subgraph(*, checkpointer: Any = None):
    """Build the checkpointable RAG design and review loop."""
    graph = StateGraph(
        RagDesignState,
        input_schema=RagDesignInputState,
        output_schema=RagDesignOutputState,
    )
    graph.add_node("rag_prepare", _rag_prepare_node)
    graph.add_node("rag_design_task", _rag_design_task_node)
    graph.add_node("design_review", _design_review_node)
    graph.add_edge(START, "rag_prepare")
    graph.add_edge("rag_prepare", "rag_design_task")
    graph.add_conditional_edges(
        "rag_design_task",
        lambda state: "rag_design_task" if state.get("rag_task_queue") else "design_review",
        {"rag_design_task": "rag_design_task", "design_review": "design_review"},
    )
    graph.add_conditional_edges(
        "design_review",
        lambda state: END if state.get("designs_confirmed") else "rag_design_task",
        {END: END, "rag_design_task": "rag_design_task"},
    )
    return graph.compile(checkpointer=checkpointer, name="rag_designs")
