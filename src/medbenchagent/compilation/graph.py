"""Serial Supervisor coordination using LangGraph's own state and persistence."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import TypeAdapter

from medbenchagent.runtime.construction import record_event

from .artifacts import ArtifactIntegrityError, _within, register_file
from . import interfaces
from .contracts import (
    ArtifactRef, AskUser, Delegate, Finish, Identifier, SupervisorDecision, Text,
    WorkIssue, WorkRecord, WorkRequest, WorkResult, Workstream,
)


class CompilationState(TypedDict, total=False):
    compilation_id: str
    goal: str
    sequence: int
    request: dict[str, Any]
    result: dict[str, Any]
    artifacts: list[dict[str, Any]]
    workstreams: list[dict[str, Any]]
    history: list[dict[str, Any]]
    answers: list[str]
    question: str | None
    next_node: str
    status: Literal["running", "waiting", "stopped"]
    goal_result: str
    feedback: list[str]
    problem_attempts: dict[str, int]
    total_repairs: int
    no_progress: int
    summary: str


_decision = TypeAdapter(SupervisorDecision)
_identifier = TypeAdapter(Identifier)
_text = TypeAdapter(Text)


def _catalog(state: CompilationState, directory: Path) -> dict[str, ArtifactRef]:
    known = {}
    paths = {}
    for raw in state.get("artifacts", []):
        ref = ArtifactRef.model_validate(raw)
        # A missing registered input is work feedback, not a coordination failure.
        # Producers still have to supply actual files in collect().
        actual = _within(directory, ref.path)
        if ref.artifact_id in known and known[ref.artifact_id] != ref:
            raise ValueError("An Artifact ID cannot refer to different files")
        if actual in paths and paths[actual] != ref.artifact_id:
            raise ValueError("One artifact file cannot have multiple identities")
        known[ref.artifact_id] = ref
        paths[actual] = ref.artifact_id
    return known


def build_compilation_graph(
    agents: Mapping[str, interfaces.AgentSpec], *, checkpointer, default_context=None, initial=None
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

    def supervisor(state: CompilationState, config: RunnableConfig, runtime: Runtime[interfaces.CompilationContext]):
        compilation_id = _identifier.validate_python(state["compilation_id"])
        if default_context is None and config.get("configurable", {}).get("thread_id") != compilation_id:
            raise ValueError("thread_id must equal compilation_id")
        goal = _text.validate_python(state["goal"])
        if runtime.context.goal != goal:
            raise ValueError("Runtime goal must match the Compilation checkpoint")
        known = _catalog(state, runtime.context.directory)
        limits = runtime.context.repair_limits
        if (state.get("no_progress", 0) >= limits.no_progress
                or state.get("total_repairs", 0) >= limits.total
                or max(state.get("problem_attempts", {}).values(), default=0) >= limits.per_problem):
            return {"status": "stopped", "goal_result": "blocked", "next_node": END,
                    "summary": "Repair budget exhausted; valid artifacts are preserved."}
        view = interfaces.SupervisorView(
            compilation_id=compilation_id, goal=goal,
            agents={name: spec.description for name, spec in agents.items()},
            artifacts=tuple(known.values()),
            last_result=WorkResult.model_validate(state["result"]) if state.get("result") else None,
            history=tuple(WorkRecord.model_validate(row) for row in state.get("history", [])),
            answers=tuple(state.get("answers", [])),
            workstreams=tuple(Workstream.model_validate(row) for row in state.get("workstreams", [])),
            workstream_agents=tuple(name for name, spec in agents.items() if spec.uses_workstreams),
            readiness=runtime.context.readiness(tuple(known.values())) if runtime.context.readiness else {},
            feedback=tuple(state.get("feedback", [])), goal_spec=runtime.context.goal_spec,
        )
        try:
            decision = _decision.validate_python(
                RunnableLambda(runtime.context.decide).invoke(view, config=config))
        except (ValueError, RuntimeError) as exc:
            return {"status": "stopped", "goal_result": "failed", "next_node": END,
                    "summary": str(exc)}

        def rejected(exc):
            return {"feedback": [str(exc)], "next_node": "supervisor",
                    "no_progress": state.get("no_progress", 0) + 1,
                    "total_repairs": state.get("total_repairs", 0) + 1}

        if isinstance(decision, Delegate):
            if decision.agent_id not in agents:
                return rejected(f"Unregistered Agent: {decision.agent_id}")
            for ref in decision.inputs:
                if known.get(ref.artifact_id) != ref:
                    return rejected("A delegation can only consume registered artifact references")
            sequence = state.get("sequence", 0) + 1
            request_id = f"{compilation_id}-{sequence:06d}"
            view.validate_workstream(decision.agent_id, decision.workstream_id)
            workstream_id = decision.workstream_id
            workstreams = list(view.workstreams)
            if agents[decision.agent_id].uses_workstreams and workstream_id is None:
                workstream_id = f"ws-{request_id}"
                workstreams.append(Workstream(
                    workstream_id=workstream_id, agent_id=decision.agent_id,
                    objective=decision.objective, scope=decision.scope,
                ))
            request = WorkRequest(
                request_id=request_id, workstream_id=workstream_id,
                compilation_id=compilation_id, agent_id=decision.agent_id, operation=decision.operation,
                objective=decision.objective, scope=decision.scope,
                inputs=decision.inputs, constraints=decision.constraints,
            )
            try:
                spec = agents[request.agent_id]
                if request.operation not in spec.operations:
                    raise ValueError("Select a registered operation")
                if spec.check_request:
                    spec.check_request(request)
            except (ValueError, KeyError, OSError) as exc:
                return rejected(exc)
            record_event(
                runtime.context.directory / "events.jsonl", "work_delegated",
                compilation_id=compilation_id, request_id=request.request_id,
                agent_id=request.agent_id, operation=request.operation, workstream_id=workstream_id,
                input_artifact_ids=[ref.artifact_id for ref in request.inputs],
                summary=request.objective, scope=[scope.model_dump() for scope in request.scope],
            )
            return {
                "sequence": sequence, "request": request.model_dump(mode="json"),
                "workstreams": [item.model_dump(mode="json") for item in workstreams],
                "next_node": decision.agent_id, "status": "running", "feedback": [],
            }
        if isinstance(decision, AskUser):
            return {"question": decision.question, "next_node": "ask_user", "status": "waiting"}
        assert isinstance(decision, Finish)
        try:
            if runtime.context.check_goal:
                runtime.context.check_goal(decision, tuple(known.values()), view.history)
        except (ValueError, KeyError, OSError) as exc:
            return rejected(exc)
        return {"summary": decision.summary, "next_node": END, "status": "stopped",
                "goal_result": decision.goal_result, "feedback": []}

    def collect(state: CompilationState, runtime: Runtime[interfaces.CompilationContext]):
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
                if expected.digest != ref.digest:
                    raise ArtifactIntegrityError(f"Artifact content changed: {ref.artifact_id}")
                if expected != ref:
                    raise ValueError("Output artifact identity does not match its producing request")
                if ref.artifact_id in known and known[ref.artifact_id] != ref:
                    raise ArtifactIntegrityError("Cannot replace a registered artifact identity or content")
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
                issues=[WorkIssue(code="artifact_integrity" if isinstance(exc, ArtifactIntegrityError)
                                  else "invalid_handoff", message=str(exc))],
            )
        if agents[request.agent_id].check_result:
            try:
                result = agents[request.agent_id].check_result(request, result)
            except (ValueError, KeyError, OSError) as exc:
                result = result.model_copy(update={"outcome": "progress" if accepted else "failed", "issues": [
                    *result.issues, WorkIssue(code="incomplete_operation", message=str(exc))]})
        new_content = {ref.digest for ref in accepted.values()} - {ref.digest for ref in known.values()}
        known.update(accepted)
        problems = dict(state.get("problem_attempts", {}))
        # Scope object identities survive renamed requests and republished artifact IDs.
        key = request.agent_id + "/" + request.operation + "/" + ",".join(sorted(
            identifier for scope in request.scope for identifier in scope.object_ids))
        repairing = result.outcome != "completed" or bool(result.issues)
        if repairing:
            problems[key] = problems.get(key, 0) + 1
        else:
            problems.pop(key, None)
        record = WorkRecord(request=request, result=result).model_dump(mode="json")
        record_event(
            runtime.context.directory / "events.jsonl", "work_result",
            compilation_id=request.compilation_id, request_id=request.request_id,
            agent_id=request.agent_id, workstream_id=request.workstream_id,
            outcome=result.outcome, artifact_ids=[ref.artifact_id for ref in result.artifacts],
            operation=request.operation, summary=result.summary,
            issues=[issue.model_dump() for issue in result.issues], needs=[need.model_dump(mode="json") for need in result.needs],
        )
        return {
            "result": result.model_dump(mode="json"),
            "problem_attempts": problems,
            "total_repairs": state.get("total_repairs", 0) + int(repairing),
            "no_progress": 0 if new_content else state.get("no_progress", 0) + 1,
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

    builder = StateGraph(CompilationState, context_schema=None if default_context else interfaces.CompilationContext)
    def with_context(node):
        import inspect
        from dataclasses import replace
        def invoke(state, config: RunnableConfig, runtime: Runtime[interfaces.CompilationContext]):
            context = runtime.context or default_context
            if context is None:
                raise ValueError("Compilation requires its assembled runtime context")
            runtime = replace(runtime, context=context)
            if hasattr(node, "invoke"):
                return node.invoke(state, config, context=context)
            kwargs = {"runtime": runtime} if "runtime" in inspect.signature(node).parameters else {}
            if "config" in inspect.signature(node).parameters:
                kwargs["config"] = config
            return node(state, **kwargs)
        return invoke

    builder.add_node("supervisor", with_context(supervisor))
    builder.add_node("collect", with_context(collect))
    builder.add_node("ask_user", ask_user)
    for name, spec in agents.items():
        builder.add_node(name, with_context(spec.node), input_schema=interfaces.AgentInput)
        builder.add_edge(name, "collect")
    if initial is not None:
        builder.add_node("initialize", lambda state: initial if not state.get("compilation_id") else {})
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "supervisor")
    else:
        builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", lambda state: state["next_node"],
        {name: name for name in (*agents, "supervisor", "ask_user", END)},
    )
    builder.add_edge("collect", "supervisor")
    builder.add_edge("ask_user", "supervisor")
    return builder.compile(checkpointer=checkpointer, name="compilation")
