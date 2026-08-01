"""SpecificationAgent: requirement extraction and confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from medbenchagent.agents._context import human_interaction, llm_context, revision_applied
from medbenchagent.agents.contracts import (
    SPECIFICATION_AGENT,
    SpecificationAgentInputState,
    SpecificationAgentOutputState,
    SpecificationAgentState,
)
from medbenchagent.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.construction import changed_mapping_keys
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
    try:
        with llm_context(state, SPECIFICATION_AGENT, "benchmark_info_extract"):
            info = extract_benchmark_info(requirement, state.get("benchmark_info", {}))
        if missing := missing_fields(info):
            raise ValueError(f"BenchmarkInfo is missing required fields: {', '.join(missing)}")
    except Exception as exc:
        if not state.get("initial_requirement_recorded"):
            human_interaction(
                state,
                agent=SPECIFICATION_AGENT,
                stage="benchmark_info",
                user_input=requirement,
                parse_status="failed",
                error_type=type(exc).__name__,
            )
        raise
    updates: dict[str, Any] = {
        "user_requirement": requirement,
        "benchmark_info": info,
        "benchmark_confirmed": False,
    }
    if not state.get("initial_requirement_recorded"):
        human_interaction(
            state,
            agent=SPECIFICATION_AGENT,
            stage="benchmark_info",
            user_input=requirement,
            parse_status="success",
            action="submit",
        )
        updates["initial_requirement_recorded"] = True
    if interaction_id := state.get("pending_benchmark_interaction_id"):
        revision_applied(
            state,
            agent=SPECIFICATION_AGENT,
            stage="benchmark_info",
            interaction_id=interaction_id,
            modified_object_ids=changed_mapping_keys(state.get("pending_benchmark_info", {}), info),
        )
        updates["pending_benchmark_interaction_id"] = ""
        updates["pending_benchmark_info"] = {}
    return updates


def _benchmark_confirm_node(state: SpecificationAgentState) -> dict[str, Any]:
    """Interpret a natural-language confirmation or correction for BenchmarkInfo."""
    answer = interrupt(
        {
            "kind": "benchmark_info_confirmation",
            "benchmark_info": state["benchmark_info"],
            "question": "Confirm this BenchmarkInfo, or describe the correction.",
        }
    )
    try:
        with llm_context(state, SPECIFICATION_AGENT, "benchmark_info_review"):
            action = _benchmark_review_action(answer, state["benchmark_info"])
    except Exception as exc:
        human_interaction(
            state,
            agent=SPECIFICATION_AGENT,
            stage="benchmark_info",
            user_input=answer,
            parse_status="failed",
            error_type=type(exc).__name__,
        )
        raise
    interaction_id = human_interaction(
        state,
        agent=SPECIFICATION_AGENT,
        stage="benchmark_info",
        user_input=answer,
        parse_status="success",
        action=action["action"],
    )
    if action["action"] == "confirm":
        working_blueprint_path = state["working_blueprint_path"]
        if not Path(working_blueprint_path).is_file():
            write_working_blueprint(state["benchmark_info"], working_blueprint_path)
        return {
            "benchmark_confirmed": True,
            "benchmark_feedback": "",
            "working_blueprint_path": working_blueprint_path,
        }
    return {
        "benchmark_confirmed": False,
        "benchmark_feedback": action["feedback"],
        "pending_benchmark_interaction_id": interaction_id,
        "pending_benchmark_info": state["benchmark_info"],
    }


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
