"""Interactive LangGraph orchestration for benchmark planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from medbenchagent.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.answer_balance import balance_jsonl
from medbenchagent.contracts import GenerationRequest
from medbenchagent.dataset import AnnotationValidationError, prepare_annotation_dataset
from medbenchagent.generation import generate_raw_benchmark
from medbenchagent.llm import complete_json
from medbenchagent.pdf_rag import LocalPdfRag, RagUnavailable, design_questions
from medbenchagent.task_screening import (
    apply_task_plan,
    generate_task_plan,
    revise_task_plan,
    task_plan_entries,
)

PROJECT_ROOT = Path(__file__).parents[3]


class BenchmarkState(MessagesState):
    """Only compact summaries and artifact paths are checkpointed, never table rows."""

    benchmark_info: dict[str, Any]
    user_requirement: str
    benchmark_feedback: str
    benchmark_confirmed: bool
    working_blueprint_path: str
    annotation_path: str
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    draft_task_plan: dict[str, Any]
    pending_task_feedback: dict[str, str] | None
    task_plan_confirmed: bool
    task_screening_report_path: str
    task_plans: list[dict[str, Any]]
    materials_root: str
    question_designs: list[dict[str, Any]]
    question_design_feedback: str
    designs_confirmed: bool
    generation_request: GenerationRequest
    ready_for_generation: bool
    raw_output_path: str
    benchmark_output_path: str
    generated_item_count: int
    generated_task_counts: dict[str, int]
    item_count: int
    task_counts: dict[str, int]
    seed: int
    balance_answer_positions: bool


def _latest_user_text(state: BenchmarkState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _confirmed(answer: object) -> bool:
    return str(answer).strip().lower() in {"y", "yes", "confirm", "confirmed", "确认", "是", "同意"}


def _benchmark_extract_node(state: BenchmarkState) -> dict[str, Any]:
    """Compute BenchmarkInfo; user confirmation is deliberately a separate node."""
    requirement = state.get("user_requirement") or _latest_user_text(state)
    if not requirement:
        raise ValueError("A benchmark requirement is required before BenchmarkInfo extraction.")
    if feedback := state.get("benchmark_feedback"):
        requirement = f"{requirement}\nRequested correction: {feedback}"
    info = extract_benchmark_info(requirement, state.get("benchmark_info", {}))
    if missing := missing_fields(info):
        raise ValueError(f"BenchmarkInfo is missing required fields: {', '.join(missing)}")
    return {"user_requirement": requirement, "benchmark_info": info, "benchmark_confirmed": False}


def _benchmark_confirm_node(state: BenchmarkState) -> dict[str, Any]:
    """Interpret a natural-language confirmation or correction for BenchmarkInfo."""
    answer = interrupt(
        {
            "kind": "benchmark_info_confirmation",
            "benchmark_info": state["benchmark_info"],
            "question": "Confirm this BenchmarkInfo, or describe the correction.",
        }
    )
    action = _benchmark_review_action(answer, state["benchmark_info"])
    if action["action"] == "confirm":
        working_blueprint_path = state.get("working_blueprint_path") or write_working_blueprint(
            state["benchmark_info"], PROJECT_ROOT / "benchmarks" / "runs"
        )
        return {
            "benchmark_confirmed": True,
            "benchmark_feedback": "",
            "working_blueprint_path": working_blueprint_path,
        }
    return {"benchmark_confirmed": False, "benchmark_feedback": action["feedback"]}


def _benchmark_review_action(answer: object, benchmark_info: dict[str, Any]) -> dict[str, str]:
    action = complete_json(
        "Interpret a BenchmarkInfo review response. Return JSON only and do not change values yourself.",
        "Convert the user's response into exactly one of these shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","feedback":"the requested correction"}\n'
        "Confirm only when the user accepts the complete BenchmarkInfo. For revise, preserve every requested "
        "correction in feedback.\n"
        f"Current BenchmarkInfo: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"User response: {answer}",
    )
    if action == {"action": "confirm"}:
        return {"action": "confirm"}
    if (
        set(action) == {"action", "feedback"}
        and action.get("action") == "revise"
        and isinstance(action.get("feedback"), str)
        and action["feedback"].strip()
    ):
        return {"action": "revise", "feedback": action["feedback"].strip()}
    raise ValueError("BenchmarkInfo review response could not be mapped to one valid confirm or revise action.")


def _dataset_profile_node(state: BenchmarkState) -> dict[str, Any]:
    annotation_path = state.get("annotation_path")
    if not annotation_path:
        annotation_path = str(
            interrupt({"kind": "annotation_path", "question": "Provide the absolute path to annotation.jsonl."})
        ).strip()
    try:
        profile, manifest = prepare_annotation_dataset(
            annotation_path, PROJECT_ROOT / "benchmarks" / "dataset_cache"
        )
    except (AnnotationValidationError, ValueError) as exc:
        answer = interrupt(
            {"kind": "dataset_error", "message": str(exc), "question": "Provide a corrected annotation.jsonl path."}
        )
        return {"annotation_path": str(answer).strip()}
    return {
        "annotation_path": annotation_path,
        "dataset_profile": profile,
        "dataset_access_manifest": manifest,
        "task_plan_confirmed": False,
    }


def _task_plan_generate_node(state: BenchmarkState) -> dict[str, Any]:
    """Generate all decisions once; this model node never interrupts or edits Blueprint."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    return {
        "draft_task_plan": generate_task_plan(blueprint, state["dataset_profile"]),
        "pending_task_feedback": None,
        "task_plan_confirmed": False,
    }


def _review_action(answer: object, task_ids: list[str]) -> dict[str, str]:
    action = complete_json(
        "Interpret a Task Plan review response. Return JSON only. Do not make task decisions yourself.",
        "Convert the user's response into exactly one of these shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","task_id":"one exact available task id","feedback":"the requested change"}\n'
        "Confirm only when the user accepts the complete Task Plan. A revise action may target only one task. "
        "Preserve the user's requested change in feedback. "
        f"Available task ids: {json.dumps(task_ids, ensure_ascii=False)}\n"
        f"User response: {answer}",
    )
    if action == {"action": "confirm"}:
        return {"action": "confirm"}
    if (
        set(action) == {"action", "task_id", "feedback"}
        and action.get("action") == "revise"
        and isinstance(action.get("task_id"), str)
        and action["task_id"].strip() in task_ids
        and isinstance(action.get("feedback"), str)
        and action["feedback"].strip()
    ):
        return {
            "action": "revise",
            "task_id": action["task_id"].strip(),
            "feedback": action["feedback"].strip(),
        }
    raise ValueError("Task Plan review response could not be mapped to one valid confirm or revise action.")


def _task_plan_review_node(state: BenchmarkState) -> dict[str, Any]:
    """Receive one review action, interpreting natural language only when needed."""
    answer = interrupt(
        {
            "kind": "task_plan_review",
            "dataset_profile": state["dataset_profile"],
            "task_plan": state["draft_task_plan"],
            "question": "Confirm the Task Plan, or describe one task that should be revised and how.",
        }
    )
    task_ids = list(state["draft_task_plan"]["tasks"])
    action = _review_action(answer, task_ids)
    if action["action"] == "confirm":
        return {"task_plan_confirmed": True, "pending_task_feedback": None}
    return {"task_plan_confirmed": False, "pending_task_feedback": action}


def _task_plan_revise_node(state: BenchmarkState) -> dict[str, Any]:
    """Replace one reviewed entry; all other draft entries are retained verbatim."""
    feedback = state.get("pending_task_feedback")
    if not feedback:
        raise ValueError("Task Plan revision requires pending task feedback.")
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    return {
        "draft_task_plan": revise_task_plan(
            blueprint,
            state["dataset_profile"],
            state["draft_task_plan"],
            feedback["task_id"],
            feedback["feedback"],
        ),
        "pending_task_feedback": None,
        "task_plan_confirmed": False,
    }


def _apply_task_plan_node(state: BenchmarkState) -> dict[str, Any]:
    """Mutate the working Blueprint only after explicit Task Plan confirmation."""
    report_path = apply_task_plan(
        state["working_blueprint_path"], state["draft_task_plan"], state["dataset_profile"]
    )
    return {
        "task_screening_report_path": report_path,
        "task_plans": task_plan_entries(state["draft_task_plan"]),
    }


def _materials_root_node(state: BenchmarkState) -> dict[str, Any]:
    if state.get("materials_root"):
        return {}
    answer = interrupt({"kind": "materials_root", "question": "Provide the local directory containing PDF medical materials."})
    return {"materials_root": str(answer).strip()}


def _rag_design_node(state: BenchmarkState) -> dict[str, Any]:
    """Compute RAG question designs; user confirmation is deliberately a separate node."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    retained = [
        {"task_id": plan["task_id"], **blueprint["tasks"][plan["task_id"]], **plan}
        for plan in state["task_plans"]
        if plan["decision"] == "retain"
    ]
    if not retained:
        raise ValueError("Task Plan must retain at least one task before RAG design.")
    try:
        rag = LocalPdfRag(state["materials_root"], PROJECT_ROOT / "benchmarks" / "rag_cache")
    except RagUnavailable:
        raise
    return {
        "question_designs": design_questions(
            rag,
            state["benchmark_info"],
            retained,
            state.get("question_design_feedback", ""),
        ),
        "designs_confirmed": False,
    }


def _design_review_node(state: BenchmarkState) -> dict[str, Any]:
    answer = interrupt(
        {
            "kind": "question_design_confirmation",
            "question_designs": state["question_designs"],
            "question": "Confirm the question designs or describe a correction.",
        }
    )
    if _confirmed(answer):
        return {"designs_confirmed": True, "question_design_feedback": ""}
    return {"designs_confirmed": False, "question_design_feedback": str(answer)}


def _generation_request_node(state: BenchmarkState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    designs = {str(design.get("task_id")): design for design in state["question_designs"]}
    task_specs: list[dict[str, Any]] = []
    for plan in state["task_plans"]:
        if plan["decision"] != "retain":
            continue
        task = blueprint["tasks"].get(plan["task_id"], {})
        design = designs.get(plan["task_id"])
        if design is None:
            raise ValueError(f"Question Design is missing retained task: {plan['task_id']}")
        spec = {
            "task_id": plan["task_id"],
            "task_type": task.get("task_type", ""),
            "response_format": task.get("response_format", ""),
            "description": task.get("description", ""),
            "question_templates": design.get("question_templates", []),
            "placeholder_bindings": design.get("placeholder_bindings", {}),
            "validation_rules": design.get("validation_rules", []),
            "citations": design.get("citations", []),
        }
        for source, destination in (
            ("answer_list", "options"),
            ("answer_field", "answer_field"),
            ("answer_value_mapping", "answer_value_mapping"),
            ("instance_path", "instance_path"),
            ("target_text_field", "target_text_field"),
            ("bbox_field", "bbox_field"),
            ("bbox_format", "bbox_format"),
        ):
            if source in design:
                spec[destination] = design[source]
        task_specs.append(spec)
    request: GenerationRequest = {
        "benchmark_info": state["benchmark_info"],
        "dataset_access_manifest": state["dataset_access_manifest"],
        "approved_task_plans": [plan for plan in state["task_plans"] if plan["decision"] == "retain"],
        "question_designs": state["question_designs"],
        "task_specs": task_specs,
        "target_sample_count": int(state["benchmark_info"]["sample_count"]),
        "seed": int(state.get("seed", 0)),
        "balance_answer_positions": bool(state.get("balance_answer_positions", True)),
        "output_directory": str(Path(state["working_blueprint_path"]).parent),
    }
    return {"generation_request": request, "ready_for_generation": True}


def _generate_raw_benchmark_node(state: BenchmarkState) -> dict[str, Any]:
    result = generate_raw_benchmark(state["generation_request"])
    return {
        "raw_output_path": result.raw_output_path,
        "generated_item_count": result.item_count,
        "generated_task_counts": result.task_counts,
        "item_count": result.item_count,
        "task_counts": result.task_counts,
    }


def _finalize_benchmark_node(state: BenchmarkState) -> dict[str, Any]:
    raw_path = state["raw_output_path"]
    if not state["generation_request"]["balance_answer_positions"]:
        return {"benchmark_output_path": raw_path}
    output_path = str(Path(raw_path).with_name("benchmark_items.jsonl"))
    return {"benchmark_output_path": balance_jsonl(raw_path, output_path, state["generation_request"]["seed"])}


def compile_benchmark_graph(*, checkpointer: Any = None):
    """Build the interactive planning and benchmark generation workflow."""
    graph = StateGraph(BenchmarkState)
    graph.add_node("benchmark_extract", _benchmark_extract_node)
    graph.add_node("benchmark_confirm", _benchmark_confirm_node)
    graph.add_node("dataset_profile", _dataset_profile_node)
    graph.add_node("task_plan_generate", _task_plan_generate_node)
    graph.add_node("task_plan_review", _task_plan_review_node)
    graph.add_node("task_plan_revise", _task_plan_revise_node)
    graph.add_node("apply_task_plan", _apply_task_plan_node)
    graph.add_node("materials_root", _materials_root_node)
    graph.add_node("rag_design", _rag_design_node)
    graph.add_node("design_review", _design_review_node)
    graph.add_node("generation_request", _generation_request_node)
    graph.add_node("generate_raw_benchmark", _generate_raw_benchmark_node)
    graph.add_node("finalize_benchmark", _finalize_benchmark_node)
    graph.add_edge(START, "benchmark_extract")
    graph.add_edge("benchmark_extract", "benchmark_confirm")
    graph.add_conditional_edges(
        "benchmark_confirm",
        lambda state: "dataset_profile" if state.get("benchmark_confirmed") else "benchmark_extract",
        {"dataset_profile": "dataset_profile", "benchmark_extract": "benchmark_extract"},
    )
    graph.add_conditional_edges(
        "dataset_profile",
        lambda state: "task_plan_generate" if state.get("dataset_profile") else "dataset_profile",
        {"task_plan_generate": "task_plan_generate", "dataset_profile": "dataset_profile"},
    )
    graph.add_edge("task_plan_generate", "task_plan_review")
    graph.add_conditional_edges(
        "task_plan_review",
        lambda state: "apply_task_plan" if state.get("task_plan_confirmed") else "task_plan_revise",
        {"apply_task_plan": "apply_task_plan", "task_plan_revise": "task_plan_revise"},
    )
    graph.add_edge("task_plan_revise", "task_plan_review")
    graph.add_edge("apply_task_plan", "materials_root")
    graph.add_edge("materials_root", "rag_design")
    graph.add_edge("rag_design", "design_review")
    graph.add_conditional_edges(
        "design_review",
        lambda state: "generation_request" if state.get("designs_confirmed") else "rag_design",
        {"generation_request": "generation_request", "rag_design": "rag_design"},
    )
    graph.add_edge("generation_request", "generate_raw_benchmark")
    graph.add_edge("generate_raw_benchmark", "finalize_benchmark")
    graph.add_edge("finalize_benchmark", END)
    return graph.compile(checkpointer=checkpointer)


benchmark_graph = compile_benchmark_graph()
