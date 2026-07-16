"""Interactive LangGraph orchestration for benchmark planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from medbenchagent.benchmark_info import (
    extract_benchmark_field,
    extract_benchmark_info,
    merge_info,
    missing_fields,
    write_working_blueprint,
)
from medbenchagent.contracts import GenerationRequest
from medbenchagent.dataset import AnnotationValidationError, prepare_annotation_dataset
from medbenchagent.llm import ModelUnavailable
from medbenchagent.pdf_rag import LocalPdfRag, RagUnavailable, design_questions
from medbenchagent.task_screening import (
    TaskScreeningReviewRequired,
    apply_task_screening,
    screen_tasks,
)

PROJECT_ROOT = Path(__file__).parents[3]


class BenchmarkState(MessagesState):
    """Only compact summaries and artifact paths are checkpointed, never table rows."""

    benchmark_info: dict[str, Any]
    user_requirement: str
    benchmark_confirmed: bool
    working_blueprint_path: str
    annotation_path: str
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    task_screening_report_path: str
    task_screening_overrides: dict[str, dict[str, Any]]
    task_plans: list[dict[str, Any]]
    task_plans_confirmed: bool
    materials_root: str
    question_designs: list[dict[str, Any]]
    designs_confirmed: bool
    question_design_feedback: str
    generation_request: GenerationRequest
    ready_for_generation: bool


def _latest_user_text(state: BenchmarkState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _confirmed(answer: object) -> bool:
    return str(answer).strip().lower() in {"y", "yes", "confirm", "confirmed", "确认", "是", "同意"}


def _benchmark_info_node(state: BenchmarkState) -> dict[str, Any]:
    info = dict(state.get("benchmark_info", {}))
    requirement = state.get("user_requirement") or _latest_user_text(state)
    if not requirement:
        requirement = str(interrupt({"kind": "benchmark_requirement", "question": "Please describe the benchmark you need."}))
    while True:
        try:
            info = extract_benchmark_info(requirement, info)
        except ModelUnavailable as exc:
            interrupt({"kind": "model_configuration", "message": str(exc)})
            continue
        while missing := missing_fields(info):
            field = missing[0]
            answer = interrupt(
                {"kind": "benchmark_info_missing", "field": field, "question": f"Please provide {field}."}
            )
            try:
                value = extract_benchmark_field(field, str(answer))
            except ModelUnavailable as exc:
                interrupt({"kind": "model_configuration", "message": str(exc)})
                continue
            info = merge_info(info, {field: value})
        answer = interrupt(
            {
                "kind": "benchmark_info_confirmation",
                "benchmark_info": info,
                "question": "Confirm this BenchmarkInfo, or describe the correction.",
            }
        )
        if _confirmed(answer):
            working_blueprint_path = state.get("working_blueprint_path") or write_working_blueprint(
                info, PROJECT_ROOT / "benchmarks" / "runs"
            )
            return {
                "benchmark_info": info,
                "user_requirement": requirement,
                "benchmark_confirmed": True,
                "working_blueprint_path": working_blueprint_path,
            }
        requirement = f"{requirement}\nRequested correction: {answer}"
        info = merge_info(info, {})


def _dataset_node(state: BenchmarkState) -> dict[str, Any]:
    annotation_path = state.get("annotation_path")
    if not annotation_path:
        annotation_path = str(
            interrupt(
                {
                    "kind": "annotation_path",
                    "question": "Provide the absolute path to annotation.jsonl.",
                }
            )
        ).strip()
    try:
        profile, manifest = prepare_annotation_dataset(
            annotation_path, PROJECT_ROOT / "benchmarks" / "dataset_cache"
        )
        blueprint_path = state["working_blueprint_path"]
        blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
        plans = screen_tasks(blueprint, profile, state.get("task_screening_overrides"))
        report_path = apply_task_screening(blueprint_path, plans)
    except TaskScreeningReviewRequired as exc:
        answer = interrupt(
            {
                "kind": "task_support_review",
                "task_id": exc.task_id,
                "model_response": exc.response,
                "valid_annotation_paths": exc.valid_paths,
                "question": "Provide a JSON task decision with task_id, decision, supporting_fields, and reason.",
            }
        )
        try:
            override = json.loads(str(answer))
        except json.JSONDecodeError as parse_error:
            raise ValueError("Task review response must be valid JSON.") from parse_error
        return {
            "annotation_path": annotation_path,
            "task_screening_overrides": {**state.get("task_screening_overrides", {}), exc.task_id: override},
        }
    except ModelUnavailable as exc:
        interrupt({"kind": "model_configuration", "message": str(exc)})
        return {"annotation_path": annotation_path, "task_plans_confirmed": False}
    except (AnnotationValidationError, ValueError) as exc:
        answer = interrupt({"kind": "dataset_error", "message": str(exc), "question": "Provide a corrected annotation.jsonl path."})
        return {"annotation_path": str(answer).strip(), "task_screening_overrides": {}}
    answer = interrupt(
        {
            "kind": "task_plan_confirmation",
            "dataset_profile": profile,
            "dataset_access_manifest": manifest,
            "task_plans": plans,
            "question": "Confirm the task retain/drop decisions and supporting fields.",
        }
    )
    if not _confirmed(answer):
        override_answer = interrupt(
            {
                "kind": "task_plan_change_required",
                "question": "Provide a JSON task decision with task_id, decision, supporting_fields, and reason.",
            }
        )
        try:
            override = json.loads(str(override_answer))
        except json.JSONDecodeError as parse_error:
            raise ValueError("Task-plan change must be valid JSON.") from parse_error
        task_id = override.get("task_id")
        if not isinstance(task_id, str):
            raise ValueError("Task-plan change must include task_id.")
        return {
            "annotation_path": annotation_path,
            "task_screening_overrides": {**state.get("task_screening_overrides", {}), task_id: override},
        }
    return {
        "annotation_path": annotation_path,
        "dataset_profile": profile,
        "dataset_access_manifest": manifest,
        "task_screening_report_path": report_path,
        "task_plans": plans,
        "task_plans_confirmed": True,
    }


def _rag_node(state: BenchmarkState) -> dict[str, Any]:
    materials_root = state.get("materials_root")
    if not materials_root:
        materials_root = str(
            interrupt({"kind": "materials_root", "question": "Provide the local directory containing PDF medical materials."})
        ).strip()
    retained = [plan for plan in state["task_plans"] if plan["decision"] == "retain"]
    try:
        rag = LocalPdfRag(materials_root, PROJECT_ROOT / "benchmarks" / "rag_cache")
        designs = design_questions(rag, state["benchmark_info"], retained, state.get("question_design_feedback", ""))
    except ModelUnavailable as exc:
        interrupt({"kind": "model_configuration", "message": str(exc)})
        return {"materials_root": materials_root, "designs_confirmed": False}
    except RagUnavailable as exc:
        answer = interrupt({"kind": "rag_unavailable", "message": str(exc), "question": "Correct the configuration or materials directory."})
        return {"materials_root": str(answer).strip()}
    answer = interrupt(
        {"kind": "question_design_confirmation", "question_designs": designs, "question": "Confirm the question designs or describe a correction."}
    )
    if not _confirmed(answer):
        return {"materials_root": materials_root, "question_design_feedback": str(answer), "designs_confirmed": False}
    return {"materials_root": materials_root, "question_designs": designs, "designs_confirmed": True}


def _generation_request_node(state: BenchmarkState) -> dict[str, Any]:
    request: GenerationRequest = {
        "benchmark_info": state["benchmark_info"],
        "dataset_access_manifest": state["dataset_access_manifest"],
        "approved_task_plans": [plan for plan in state["task_plans"] if plan["decision"] == "retain"],
        "question_designs": state["question_designs"],
        "target_sample_count": int(state["benchmark_info"]["sample_count"]),
        "seed": 0,
    }
    return {"generation_request": request, "ready_for_generation": True}


def compile_benchmark_graph(*, checkpointer: Any = None):
    """Build the three interactive planning stages and stop at ``GenerationRequest``."""
    graph = StateGraph(BenchmarkState)
    graph.add_node("benchmark_info", _benchmark_info_node)
    graph.add_node("dataset_exploration", _dataset_node)
    graph.add_node("pdf_rag_design", _rag_node)
    graph.add_node("generation_request", _generation_request_node)
    graph.add_edge(START, "benchmark_info")
    graph.add_edge("benchmark_info", "dataset_exploration")
    graph.add_conditional_edges(
        "dataset_exploration",
        lambda state: "pdf_rag_design" if state.get("task_plans_confirmed") else "dataset_exploration",
    )
    graph.add_conditional_edges(
        "pdf_rag_design",
        lambda state: "generation_request" if state.get("designs_confirmed") else "pdf_rag_design",
    )
    graph.add_edge("generation_request", END)
    return graph.compile(checkpointer=checkpointer)


benchmark_graph = compile_benchmark_graph()
