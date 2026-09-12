"""Small checkpointable JSON-action loop shared by the benchmark agents."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from medbenchagent.construction import event_context, record_event
from medbenchagent.llm import ModelUnavailable, complete_json
from medbenchagent.run_artifacts import atomic_write_json
from medbenchagent.tool_contracts import AskUserArguments, NoArguments


class AgentState(TypedDict, total=False):
    context: dict[str, Any]
    action: dict[str, Any]
    result: dict[str, Any]
    step: int
    elapsed: float
    failures: int
    attempts: dict[str, int]
    observations: list[dict[str, Any]]


Tool = Callable[..., dict[str, Any]]


def result(status: str, summary: str, **updates: Any) -> dict[str, Any]:
    return {"status": status, "summary": summary, "updates": updates}


def compile_agent(
    name: str,
    instructions: str,
    tools: dict[str, Tool],
    describe: Callable[[dict[str, Any]], dict[str, Any]],
    finish: Tool,
    *,
    args_schemas: dict[str, type[BaseModel]] | None = None,
    human: bool = False,
    checkpointer: Any = None,
):
    """Run registered tools serially, retaining private state and completed operations."""
    schemas = {key: NoArguments for key in (*tools, "finish")}
    schemas.update(args_schemas or {})
    schemas["ask_user"] = AskUserArguments
    catalog = {
        key: {
            "description": function.__doc__ or f"Complete {name}",
            "parameters": schemas[key].model_json_schema(),
        }
        for key, function in {**tools, "finish": finish}.items()
    }
    catalog["ask_user"] = {
        "description": "Ask for clarification of unresolved requirements or source semantics.",
        "parameters": AskUserArguments.model_json_schema(),
    }

    def initialize(state: AgentState) -> dict:
        return {
            "step": 0,
            "elapsed": 0.0,
            "failures": 0,
            "attempts": {},
            "observations": [],
            "result": result("running", "Ready"),
        }

    def decide(state: AgentState) -> dict:
        context = state["context"]
        limits = context.get("limits", {})
        if (
            state["step"] >= limits.get("max_calls", 60)
            or state["elapsed"] >= limits.get("max_seconds", 3600)
            or state["failures"] >= limits.get("max_failures", 3)
        ):
            return {
                "result": result(
                    "blocked", "Execution budget exhausted; inspect the last tool result."
                )
            }
        started = time.monotonic()
        with event_context(
            construction_events_path=context["construction_events_path"],
            run_id=context["run_id"],
            agent=name,
            operation="decide",
        ):
            try:
                action = complete_json(
                    instructions
                    + '\nReturn exactly {"tool":"name","arguments":{},"reason":"brief reason"}. '
                    'Choose a registered tool, finish, or ask_user according to its JSON Schema. '
                    "Do not invent source facts. Do not repeat an unsuccessful action unchanged. "
                    "Use finish only when all required outputs exist. Tools execute serially.",
                    json.dumps(
                        {
                            "tools": catalog,
                            "state": describe(context),
                            "recent_observations": state.get("observations", []),
                            "last_result": {
                                k: v for k, v in state["result"].items() if k != "updates"
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
                if (
                    set(action) != {"tool", "arguments", "reason"}
                    or not isinstance(action["tool"], str)
                    or action["tool"] not in {*tools, "finish", "ask_user"}
                    or not isinstance(action["arguments"], dict)
                    or not isinstance(action["reason"], str)
                ):
                    raise ValueError("Invalid tool action")
            except (ModelUnavailable, ValueError) as exc:
                return {
                    "result": result("failed", str(exc)),
                    "failures": state["failures"] + 1,
                    "step": state["step"] + 1,
                    "action": {},
                    "elapsed": state["elapsed"] + time.monotonic() - started,
                }
        return {"action": action, "elapsed": state["elapsed"] + time.monotonic() - started}

    def execute(state: AgentState) -> dict:
        started = time.monotonic()
        action, context = state["action"], copy.deepcopy(state["context"])
        tool_name, arguments = action["tool"], action["arguments"]
        step = state["step"] + 1
        attempts = dict(state.get("attempts", {}))
        identity = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=False)
        attempts[identity] = attempts.get(identity, 0) + 1
        if attempts[identity] > context.get("limits", {}).get("max_reworks", 3) + 1:
            return {
                "result": result("blocked", "Repeated action exceeded the rework budget"),
                "step": step,
            }
        operation = Path(context["operation_directory"]) / name / f"{step}.json"
        try:
            arguments = schemas[tool_name].model_validate(arguments).model_dump()
            parameter_error = None
        except ValidationError as exc:
            parameter_error = [
                {"field": ".".join(map(str, error["loc"])), "message": error["msg"]}
                for error in exc.errors(include_url=False, include_input=False)
            ]
        if parameter_error is not None:
            outcome = result("failed", "Invalid tool arguments")
            outcome["errors"] = parameter_error
        elif operation.is_file():
            saved = json.loads(operation.read_text(encoding="utf-8"))
            if saved["action"] != action:
                raise ValueError("Operation identity was reused for a different action")
            outcome = saved["result"]
        elif tool_name == "ask_user":
            question = arguments["question"]
            if human:
                answer = interrupt({"kind": "clarification", "question": question})
                if not isinstance(answer, str) or not answer.strip():
                    outcome = result("failed", "A natural-language answer is required")
                else:
                    outcome = result(
                        "success",
                        "User clarification received",
                        user_requirement=context["user_requirement"] + "\n" + answer,
                    )
            else:
                outcome = result("needs_input", question)
        else:
            tool_context = dict(context, operation_directory=str(operation.with_suffix("")))
            with event_context(
                construction_events_path=context["construction_events_path"],
                run_id=context["run_id"],
                agent=name,
                operation=tool_name,
            ):
                try:
                    function = finish if tool_name == "finish" else tools[tool_name]
                    tool = StructuredTool.from_function(
                        func=lambda **kwargs: function(tool_context, **kwargs),
                        name=tool_name,
                        description=catalog[tool_name]["description"],
                        args_schema=schemas[tool_name],
                    )
                    outcome = tool.invoke(arguments)
                except (ValueError, OSError, ModelUnavailable) as exc:
                    outcome = result("failed", f"{type(exc).__name__}: {exc}")
        context.update(outcome.get("updates", {}))
        if not operation.is_file():
            atomic_write_json(operation, {"action": action, "result": outcome})
            record_event(
                context["construction_events_path"],
                "tool_result",
                run_id=context["run_id"],
                agent=name,
                operation=tool_name,
                status=outcome["status"],
                reason=action["reason"],
                summary=outcome["summary"],
            )
        observations = [
            *state.get("observations", []),
            {"tool": tool_name, "status": outcome["status"], "summary": outcome["summary"][:6000]},
        ][-12:]
        return {
            "context": context,
            "result": outcome,
            "step": step,
            "attempts": attempts,
            "observations": observations,
            "elapsed": state["elapsed"] + time.monotonic() - started,
            "failures": state["failures"] + 1 if outcome["status"] == "failed" else 0,
        }

    graph = StateGraph(AgentState)
    graph.add_node("initialize", initialize)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda s: (
            END
            if s["result"]["status"] == "blocked"
            else "execute"
            if s.get("action")
            else "decide"
        ),
    )
    graph.add_conditional_edges(
        "execute",
        lambda s: (
            END if s["result"]["status"] in {"complete", "needs_input", "blocked"} else "decide"
        ),
    )
    return graph.compile(checkpointer=checkpointer, name=name)
