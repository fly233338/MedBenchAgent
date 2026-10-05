"""The host-orchestrated OpenHands dataset preparation agent.

This module is the preparation entrypoint.  OpenHands owns exploration, mapping,
code, execution and repair; the host state machine owns phase boundaries.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medbenchagent.runtime.preparation_orchestrator import run_preparation_orchestrator
from medbenchagent.runtime.construction import record_event


class DatasetAgentState(TypedDict, total=False):
    context: dict
    result: dict


DATASET_PROCESSING_AGENT = "DatasetProcessingAgent"


def _dataset_event(context: dict, event_type: str, **values: object) -> None:
    path = context.get("construction_events_path")
    if not path:
        return
    payload = {"run_id": context.get("run_id"), "agent": DATASET_PROCESSING_AGENT,
               "dataset_id": context.get("dataset_scope"),
               "conversation_id": context.get("preparation_conversation_id"),
               "workspace": context.get("preparation_workspace")}
    payload.update(values)
    record_event(path, event_type, **payload)


def run_dataset_agent(context: dict):
    context = dict(context)
    _dataset_event(context, "agent_started", operation="prepare_data", status="started",
                   source_count=len(context.get("sources", [])))
    try:
        outcome = run_preparation_orchestrator(context)
    except ValueError as exc:
        outcome = {"status": "blocked", "summary": str(exc), "updates": {}}
    updates = outcome.get("updates", {})
    _dataset_event({**context, **updates}, "worker_result", operation="preparation_orchestrator",
                   status=outcome["status"], diagnostic=outcome.get("summary", ""),
                   termination_reason=updates.get("preparation_termination_reason"))
    _dataset_event({**context, **updates}, "agent_finished", operation="prepare_data",
                   status=outcome["status"], diagnostic=outcome.get("summary", ""),
                   candidate_path=(updates.get("preparation_candidate") or {}).get("candidate_path"))
    return outcome


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



