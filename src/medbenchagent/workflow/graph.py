"""Interactive LangGraph orchestration for benchmark planning."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from medbenchagent.benchmark_info import extract_benchmark_info, merge_info, missing_fields, write_working_blueprint
from medbenchagent.contracts import GenerationRequest
from medbenchagent.dataset import build_access_layer, discover_dataset, infer_schema, plan_task_capabilities, propose_schema
from medbenchagent.llm import ModelUnavailable
from medbenchagent.pdf_rag import LocalPdfRag, RagUnavailable, design_questions

PROJECT_ROOT = Path(__file__).parents[3]
BLUEPRINT = json.loads((PROJECT_ROOT / "blueprint.json").read_text(encoding="utf-8"))


class BenchmarkState(MessagesState):
    """Only compact summaries and artifact paths are checkpointed, never table rows."""

    benchmark_info: dict[str, Any]
    user_requirement: str
    user_constraints: list[str]
    benchmark_confirmed: bool
    working_blueprint_path: str
    dataset_root: str
    dataset_inventory: dict[str, Any]
    dataset_schema_map: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    task_plans: list[dict[str, Any]]
    task_plans_confirmed: bool
    task_plan_feedback: str
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
        missing = missing_fields(info)
        if missing:
            field = missing[0]
            answer = interrupt(
                {"kind": "benchmark_info_missing", "field": field, "question": f"Please provide {field}."}
            )
            requirement = f"{requirement}\nSupplement for {field}: {answer}"
            continue
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


def _apply_mapping_answer(schema_map: dict[str, Any], answer: object) -> dict[str, Any]:
    """Accept explicit ``case_key=<column>`` and ``image_column=<column>`` constraints only."""
    constraints = dict(re.findall(r"\b(case_key|image_column)\s*=\s*([A-Za-z0-9_]+)", str(answer)))
    canonical = next(
        entity for entity in schema_map["entities"] if entity["table_id"] == schema_map["canonical_table"]
    )
    columns = canonical["columns"]
    for field, column in constraints.items():
        if column not in columns:
            raise ValueError(f"{column!r} is not a column in the canonical table.")
        schema_map[field] = column
    join = re.search(
        r"\bjoin\s*=\s*(table_\d+)\.([A-Za-z0-9_]+)\s*:\s*(table_\d+)\.([A-Za-z0-9_]+)", str(answer)
    )
    if join:
        left_table, left_column, right_table, right_column = join.groups()
        entities = {entity["table_id"]: entity for entity in schema_map["entities"]}
        if left_table not in entities or right_table not in entities:
            raise ValueError("The requested join references an unknown table.")
        if left_column not in entities[left_table]["columns"] or right_column not in entities[right_table]["columns"]:
            raise ValueError("The requested join references an unknown column.")
        schema_map["joins"] = [
            {
                "left_table": left_table,
                "left_column": left_column,
                "right_table": right_table,
                "right_column": right_column,
            }
        ]
    unresolved = []
    if not schema_map.get("case_key"):
        unresolved.append("Select the stable case or image identifier column using case_key=<column>.")
    if len(canonical["candidate_image_columns"]) > 1 and not schema_map.get("image_column"):
        unresolved.append("Select the image-reference column using image_column=<column>.")
    joined_tables = {item["right_table"] for item in schema_map.get("joins", [])}
    for entity in schema_map["entities"]:
        if entity["table_id"] != schema_map["canonical_table"] and entity["table_id"] not in joined_tables:
            unresolved.append(f"Confirm the join path for {entity['table_id']}.")
    schema_map["unknowns"] = unresolved
    return schema_map


def _apply_task_feedback(plans: list[dict[str, Any]], feedback: str) -> list[dict[str, Any]]:
    """Apply only explicit, auditable quota and decision modifications."""
    match = re.search(r"task_id\s*=\s*([A-Za-z0-9_]+).*?(?:count|recommended_count)\s*=\s*(\d+)", feedback)
    if match:
        task_id, count = match.groups()
        for plan in plans:
            if plan["task_id"] == task_id and plan["decision"] == "retain":
                plan["recommended_count"] = min(int(count), int(plan["eligible_case_count"]))
    drop = re.search(r"task_id\s*=\s*([A-Za-z0-9_]+).*?decision\s*=\s*drop", feedback)
    if drop:
        for plan in plans:
            if plan["task_id"] == drop.group(1):
                plan["decision"] = "drop"
                plan["recommended_count"] = 0
                plan["decision_reason"] = "Explicitly dropped by the user."
    return plans


def _dataset_node(state: BenchmarkState) -> dict[str, Any]:
    root = state.get("dataset_root")
    if not root:
        root = str(
            interrupt(
                {
                    "kind": "dataset_root",
                    "question": "Provide the local directory containing CSV/XLSX files and image assets.",
                }
            )
        ).strip()
    try:
        inventory = discover_dataset(root)
        schema_map = infer_schema(inventory)
        schema_map = propose_schema(inventory, schema_map)
        constraints = list(state.get("user_constraints", []))
        for constraint in constraints:
            schema_map = _apply_mapping_answer(schema_map, constraint)
        if schema_map["unknowns"]:
            answer = interrupt(
                {
                    "kind": "schema_mapping_review",
                    "schema_candidates": schema_map,
                    "question": schema_map["unknowns"][0],
                }
            )
            return {"dataset_root": root, "user_constraints": [*constraints, str(answer)]}
        manifest = build_access_layer(inventory, schema_map, PROJECT_ROOT / "benchmarks" / "runs")
        blueprint_path = state.get("working_blueprint_path")
        blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8")) if blueprint_path else BLUEPRINT
        plans = plan_task_capabilities(blueprint, manifest, schema_map)
        plans = _apply_task_feedback(plans, state.get("task_plan_feedback", ""))
    except ModelUnavailable as exc:
        interrupt({"kind": "model_configuration", "message": str(exc)})
        return {"dataset_root": root, "task_plans_confirmed": False}
    except ValueError as exc:
        answer = interrupt({"kind": "dataset_error", "message": str(exc), "question": "Provide a corrected path or mapping."})
        return {"dataset_root": str(answer).strip(), "user_constraints": []}
    compact_inventory = {key: value for key, value in inventory.items() if key != "_raw_tables" and key != "image_paths"}
    answer = interrupt(
        {
            "kind": "task_plan_confirmation",
            "inventory": compact_inventory,
            "schema_map": schema_map,
            "task_plans": plans,
            "question": "Confirm the task decisions and recommended counts, or provide a correction.",
        }
    )
    if not _confirmed(answer):
        return {
            "dataset_root": root,
            "user_constraints": [*constraints, str(answer)],
            "task_plan_feedback": str(answer),
            "task_plans_confirmed": False,
        }
    return {
        "dataset_root": root,
        "dataset_inventory": compact_inventory,
        "dataset_schema_map": schema_map,
        "dataset_access_manifest": manifest,
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
