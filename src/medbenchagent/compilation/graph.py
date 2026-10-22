"""Serial Supervisor coordination using LangGraph's own state and persistence."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import TypeAdapter

from .artifacts import register_file, resolve_artifact
from .contracts import (
    ArtifactRef, AskUser, Delegate, Finish, Identifier, SupervisorDecision, Text,
    WorkIssue, WorkRequest, WorkResult,
)


class AgentInput(TypedDict):
    request: dict[str, Any]


class AgentOutput(TypedDict):
    result: dict[str, Any]


class CompilationState(TypedDict, total=False):
    compilation_id: str
    goal: str
    sequence: int
    request: dict[str, Any]
    result: dict[str, Any]
    artifacts: list[dict[str, Any]]
    history: list[dict[str, str]]
    answers: list[str]
    question: str | None
    next_node: str
    status: Literal["running", "waiting", "completed"]
    summary: str


@dataclass(frozen=True)
class SupervisorView:
    compilation_id: str
    goal: str
    agents: Mapping[str, str]
    artifacts: tuple[ArtifactRef, ...]
    last_result: WorkResult | None
    history: tuple[dict[str, str], ...]
    answers: tuple[str, ...]


@dataclass(frozen=True)
class CompilationContext:
    directory: Path
    decide: Callable[[SupervisorView], SupervisorDecision]


@dataclass(frozen=True)
class AgentSpec:
    description: str
    node: Any  # A LangGraph node function or a compiled subgraph with AgentInput/AgentOutput.


_decision = TypeAdapter(SupervisorDecision)
_identifier = TypeAdapter(Identifier)
_text = TypeAdapter(Text)


def _catalog(state: CompilationState, directory: Path) -> dict[str, ArtifactRef]:
    known = {}
    paths = {}
    for raw in state.get("artifacts", []):
        ref = ArtifactRef.model_validate(raw)
        actual = resolve_artifact(directory, ref)
        if ref.artifact_id in known and known[ref.artifact_id] != ref:
            raise ValueError("An Artifact ID cannot refer to different files")
        if actual in paths and paths[actual] != ref.artifact_id:
            raise ValueError("One artifact file cannot have multiple identities")
        known[ref.artifact_id] = ref
        paths[actual] = ref.artifact_id
    return known


def build_compilation_graph(
    agents: Mapping[str, AgentSpec], *, checkpointer
):
    """Assemble named nodes; policies choose work and agents choose their persistence."""
    agents = dict(agents)
    reserved = {"supervisor", "collect", "ask_user", START, END}
    if not agents:
        raise ValueError("Register at least one professional Agent")
    for name in agents:
        _identifier.validate_python(name)
        if name in reserved:
            raise ValueError(f"Reserved Agent name: {name}")
    if checkpointer is None or isinstance(checkpointer, bool):
        raise ValueError("The parent graph needs an actual checkpointer")

    def supervisor(state: CompilationState, config: RunnableConfig, runtime: Runtime[CompilationContext]):
        compilation_id = _identifier.validate_python(state["compilation_id"])
        if config.get("configurable", {}).get("thread_id") != compilation_id:
            raise ValueError("thread_id must equal compilation_id")
        goal = _text.validate_python(state["goal"])
        known = _catalog(state, runtime.context.directory)
        view = SupervisorView(
            compilation_id=compilation_id, goal=goal,
            agents={name: spec.description for name, spec in agents.items()},
            artifacts=tuple(known.values()),
            last_result=WorkResult.model_validate(state["result"]) if state.get("result") else None,
            history=tuple(dict(row) for row in state.get("history", [])),
            answers=tuple(state.get("answers", [])),
        )
        decision = _decision.validate_python(runtime.context.decide(view))
        if isinstance(decision, Delegate):
            if decision.agent_id not in agents:
                raise ValueError(f"Unregistered Agent: {decision.agent_id}")
            for ref in decision.inputs:
                if known.get(ref.artifact_id) != ref:
                    raise ValueError("A delegation can only consume registered artifact references")
            sequence = state.get("sequence", 0) + 1
            request = WorkRequest(
                request_id=f"{compilation_id}-{sequence:06d}",
                compilation_id=compilation_id, agent_id=decision.agent_id,
                objective=decision.objective, scope=decision.scope,
                inputs=decision.inputs, constraints=decision.constraints,
            )
            return {
                "sequence": sequence, "request": request.model_dump(mode="json"),
                "next_node": decision.agent_id, "status": "running",
            }
        if isinstance(decision, AskUser):
            return {"question": decision.question, "next_node": "ask_user", "status": "waiting"}
        assert isinstance(decision, Finish)
        return {"summary": decision.summary, "next_node": END, "status": "completed"}

    def collect(state: CompilationState, runtime: Runtime[CompilationContext]):
        request = WorkRequest.model_validate(state["request"])
        known = _catalog(state, runtime.context.directory)
        try:
            result = WorkResult.model_validate(state.get("result"))
            if result.request_id != request.request_id:
                raise ValueError("WorkResult belongs to another request")
            accepted = {}
            for ref in result.artifacts:
                expected = register_file(
                    runtime.context.directory, request, ref.path, kind=ref.kind,
                )
                if expected != ref:
                    raise ValueError("Output artifact identity does not match its producing request")
                if ref.artifact_id in accepted:
                    raise ValueError("Duplicate output Artifact ID")
                accepted[ref.artifact_id] = ref
            visible = known | accepted
            for need in result.needs:
                if need.detail_ref and visible.get(need.detail_ref.artifact_id) != need.detail_ref:
                    raise ValueError("Need detail must reference a registered artifact")
        except (ValueError, OSError) as exc:
            # Invalid handoffs are feedback, not a claim of successful business work.
            accepted = {}
            result = WorkResult(
                request_id=request.request_id, outcome="failed",
                summary="Agent returned an invalid handoff",
                issues=[WorkIssue(code="invalid_handoff", message=str(exc))],
            )
        known.update(accepted)
        record = {
            "request_id": request.request_id, "agent_id": request.agent_id,
            "outcome": result.outcome, "summary": result.summary,
        }
        return {
            "result": result.model_dump(mode="json"),
            "artifacts": [ref.model_dump(mode="json") for ref in known.values()],
            "history": [*state.get("history", []), record],
        }

    def ask_user(state: CompilationState):
        answer = interrupt({
            "compilation_id": state["compilation_id"], "question": state["question"],
        })
        return {
            "answers": [*state.get("answers", []), _text.validate_python(answer)],
            "question": None, "status": "running",
        }

    builder = StateGraph(CompilationState, context_schema=CompilationContext)
    builder.add_node("supervisor", supervisor)
    builder.add_node("collect", collect)
    builder.add_node("ask_user", ask_user)
    for name, spec in agents.items():
        builder.add_node(name, spec.node, input_schema=AgentInput)
        builder.add_edge(name, "collect")
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", lambda state: state["next_node"],
        {name: name for name in (*agents, "ask_user", END)},
    )
    builder.add_edge("collect", "supervisor")
    builder.add_edge("ask_user", "supervisor")
    return builder.compile(checkpointer=checkpointer, name="compilation")
