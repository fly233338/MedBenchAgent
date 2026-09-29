"""Thin LangGraph wrapper around the shared PreparationService."""
from __future__ import annotations

from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from medbenchagent.runtime.preparation_service import PreparationService
from medbenchagent.runtime.preparation_bridge import initialize_preparation_context, apply_preparation_snapshot


class DatasetAgentState(TypedDict, total=False):
    context: dict
    result: dict


def run_dataset_agent(context: dict):
    context = dict(context)
    sources = list(context.get("sources", []))
    if not sources:
        return {"status": "blocked", "summary": "Dataset has no configured source", "updates": {}}
    workspace = context.get("preparation_workspace") or context.get("run_directory")
    context = initialize_preparation_context(context, sources=sources, workspace=workspace,
                                             goal=context.get("user_requirement", ""))
    service = context.get("preparation_service") or PreparationService(runtime=context.get("preparation_runtime"))
    ref = service.create_or_resume({"dataset_id": context.get("dataset_scope", "dataset"),
                                    "workspace": workspace, "sources": sources,
                                    "goal": context.get("user_requirement", "")})
    snapshot = service.step(ref, stop_requested=context.get("stop_requested"))
    updates = apply_preparation_snapshot(context, snapshot)
    return {"status": snapshot.get("status", "blocked"),
            "summary": snapshot.get("last_result", {}).get("diagnostic", ""),
            "updates": updates}


def compile_dataset_agent(*, checkpointer=None):
    def execute(state: DatasetAgentState):
        context = dict(state["context"])
        outcome = run_dataset_agent(context)
        context.update(outcome.get("updates", {}))
        context["dataset_agent_result"] = {k: v for k, v in outcome.items() if k != "updates"}
        return {"context": context, "result": context["dataset_agent_result"]}

    graph = StateGraph(DatasetAgentState)
    graph.add_node("dataset_processing_agent", execute)
    graph.add_edge(START, "dataset_processing_agent")
    graph.add_edge("dataset_processing_agent", END)
    return graph.compile(checkpointer=checkpointer)
