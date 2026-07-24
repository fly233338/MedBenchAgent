"""SpecificationAgent: requirement extraction and confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from medbenchagent.agents._context import human_action, llm_context
from medbenchagent.agents.contracts import (
    SPECIFICATION_AGENT,
    SpecificationAgentInputState,
    SpecificationAgentOutputState,
    SpecificationAgentState,
)
from medbenchagent.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.llm import complete_json


def _latest_user_text(state: SpecificationAgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _benchmark_extract_node(state: SpecificationAgentState) -> dict[str, Any]:
    """Compute BenchmarkInfo; user confirmation is deliberately a separate node."""
    requirement = state.get("user_requirement") or _latest_user_text(state)
    if not requirement:
        raise ValueError("A benchmark requirement is required before BenchmarkInfo extraction.")
    if feedback := state.get("benchmark_feedback"):
        requirement = f"{requirement}\nRequested correction: {feedback}"
    with llm_context(state, SPECIFICATION_AGENT, "benchmark_info_extract"):
        info = extract_benchmark_info(requirement, state.get("benchmark_info", {}))
    if missing := missing_fields(info):
        raise ValueError(f"BenchmarkInfo is missing required fields: {', '.join(missing)}")
    return {"user_requirement": requirement, "benchmark_info": info, "benchmark_confirmed": False}


def _benchmark_confirm_node(state: SpecificationAgentState) -> dict[str, Any]:
    """Interpret a natural-language confirmation or correction for BenchmarkInfo."""
    answer = interrupt(
        {
            "kind": "benchmark_info_confirmation",
            "benchmark_info": state["benchmark_info"],
            "question": "Confirm this BenchmarkInfo, or describe the correction.",
        }
    )
    with llm_context(state, SPECIFICATION_AGENT, "benchmark_info_review"):
        action = _benchmark_review_action(answer, state["benchmark_info"])
    if action["action"] == "confirm":
        working_blueprint_path = state["working_blueprint_path"]
        if not Path(working_blueprint_path).is_file():
            write_working_blueprint(state["benchmark_info"], working_blueprint_path)
        human_action(
            state,
            agent=SPECIFICATION_AGENT,
            stage="benchmark_info",
            action="confirm",
            modified_objects=0,
        )
        return {
            "benchmark_confirmed": True,
            "benchmark_feedback": "",
            "working_blueprint_path": working_blueprint_path,
        }
    human_action(
        state,
        agent=SPECIFICATION_AGENT,
        stage="benchmark_info",
        action="revise",
        modified_objects=1,
    )
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


def compile_specification_agent(*, checkpointer: Any = None):
    """Build the checkpointable SpecificationAgent confirmation loop."""
    graph = StateGraph(
        SpecificationAgentState,
        input_schema=SpecificationAgentInputState,
        output_schema=SpecificationAgentOutputState,
    )
    graph.add_node("benchmark_extract", _benchmark_extract_node)
    graph.add_node("benchmark_confirm", _benchmark_confirm_node)
    graph.add_edge(START, "benchmark_extract")
    graph.add_edge("benchmark_extract", "benchmark_confirm")
    graph.add_conditional_edges(
        "benchmark_confirm",
        lambda state: END if state.get("benchmark_confirmed") else "benchmark_extract",
        {END: END, "benchmark_extract": "benchmark_extract"},
    )
    return graph.compile(checkpointer=checkpointer, name=SPECIFICATION_AGENT)
