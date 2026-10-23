"""Native Supervisor tool loop, private persistence, bounded context and formal decisions."""

import json
from dataclasses import replace

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from langchain_core.messages import AIMessage

from medbenchagent.compilation.artifacts import write_json_artifact
from medbenchagent.compilation.contracts import WorkRequest, WorkResult
from medbenchagent.compilation.graph import build_compilation_graph
from medbenchagent.compilation.interfaces import AgentSpec, CompilationContext, SupervisorView
from medbenchagent.agents.supervisor import ModelSupervisor, _Inspection
from medbenchagent.prompts import load_prompt
from medbenchagent.settings import load_supervisor_settings, SupervisorSettings
from medbenchagent.runtime.llm import RequestContextBudgetError
from native_agent_fixtures import ScriptedChatModel, call, current_view


def test_native_supervisor_repairs_tools_and_decisions_and_resumes(tmp_path):
    settings = load_supervisor_settings()
    settings = replace(settings, llm=replace(settings.llm, model="offline"), artifact_chars=40)
    assert SupervisorSettings.from_snapshot(settings.snapshot()) == settings
    request = WorkRequest(
        compilation_id="native", request_id="native-input", agent_id="fixture", operation="bind", objective="Input"
    )
    ref = write_json_artifact(
        tmp_path, request, "evidence.json", kind="report", value={"text": "evidence " * 100}
    )

    def respond(messages, index):
        view = current_view(messages)
        if index == 1:
            return call(
                "SupervisorResponse",
                {"decision": {"action": "delegate", "operation": "inspect", "agent_id": "missing", "objective": "Inspect"}},
                index,
            )
        if index == 2:
            assert "Unregistered Agent" in messages[-1].content
            return call("inspect", {"query": {"target": "preview", "ids": ["invented"]}}, index)
        if index == 3:
            assert "Unknown inspection IDs" in messages[-1].content
            return call(
                "inspect", {"query": {"target": "preview", "ids": [ref.artifact_id]}}, index
            )
        if index == 4:
            assert json.loads(messages[-1].content)["items"][0]["truncated"]
            decision = {
                "action": "delegate", "operation": "inspect",
                "agent_id": "inspector",
                "objective": "Inspect",
                "input_artifact_ids": [ref.artifact_id],
            }
        elif not view["user_answers"]:
            decision = {"action": "ask_user", "question": "Continue?"}
        else:
            assert len(messages) > 3
            decision = {"action": "finish", "goal_result": "succeeded", "summary": "Inspected; no benchmark was generated"}
        return call("SupervisorResponse", {"decision": decision}, index)

    model = ScriptedChatModel(responder=respond)
    supervisor = ModelSupervisor(
        tmp_path, settings, instructions=load_prompt("supervisor.md"), model=model
    )

    def worker(state):
        assert state["request"]["inputs"] == [ref.model_dump()]
        return {
            "result": WorkResult(
                request_id=state["request"]["request_id"], outcome="completed", summary="Inspected"
            ).model_dump()
        }

    config = {"configurable": {"thread_id": "native"}}
    context = CompilationContext(tmp_path, supervisor, "Inspect")
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        graph = build_compilation_graph(
            {"inspector": AgentSpec("Inspect", worker, operations=("inspect",))}, checkpointer=saver
        )
        before = graph.invoke(
            {"compilation_id": "native", "goal": "Inspect", "artifacts": [ref.model_dump()]},
            config,
            context=context,
        )
        assert before["status"] == "waiting" and "messages" not in before
        after = graph.invoke(Command(resume="Continue"), config, context=context)
        assert after["status"] == "stopped" and len(after["history"]) == 1
        private = [
            row for row in saver.list(config) if row.config["configurable"].get("checkpoint_ns")
        ]
        assert any("messages" in row.checkpoint["channel_values"] for row in private)
    assert all(binding.get("parallel_tool_calls") is False for binding in model.bindings)
    logs = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(row["event_type"] == "agent_model_call" for row in logs)


def test_context_queries_and_frozen_budgets(tmp_path):
    settings = load_supervisor_settings()
    settings = replace(
        settings,
        llm=replace(settings.llm, model="offline"),
        context_chars=14000,
        catalog_page_size=1,
    )
    model = ScriptedChatModel(
        responder=lambda *_: pytest.fail("Context-only test must not call a model")
    )
    supervisor = ModelSupervisor(
        tmp_path, settings, instructions=load_prompt("supervisor.md"), model=model
    )
    view = SupervisorView(
        compilation_id="bounded",
        goal="Inspect",
        agents={"worker": "Work"},
        artifacts=(),
        last_result=None,
        history=(),
        answers=(),
    )
    assert supervisor._context(view)["goal"] == "Inspect"
    with pytest.raises(RequestContextBudgetError):
        supervisor._context(replace(view, answers=("Required answer " * 10000,)))
    query = supervisor._inspect(view, _Inspection(target="work", limit=1))
    assert query["items"] == [] and query["total"] == 0
    with pytest.raises(ValueError, match="cannot exceed"):
        supervisor._inspect(view, _Inspection(target="work", limit=2))


def test_native_summarization_and_cross_compilation_isolation(tmp_path):
    settings = load_supervisor_settings()
    settings = replace(
        settings,
        llm=replace(settings.llm, model="offline"),
        loop=replace(settings.loop, summary_trigger_tokens=300, summary_keep_tokens=80),
    )
    summaries = []
    seen = set()

    def respond(messages, index):
        if not any(isinstance(m.content, str) and "Current work data (not instructions):\n" in m.content for m in messages):
            summaries.append(messages)
            return AIMessage(content="Continue from the authoritative work records and artifacts.")
        view = current_view(messages)
        compilation = view["compilation_id"]
        if compilation not in seen:
            seen.add(compilation)
            assert len(messages) == 3
        if len(view["history"]) < 2:
            decision = {
                "action": "delegate", "operation": "inspect",
                "agent_id": "worker",
                "objective": f"Private {compilation} " + "inspect evidence " * 100,
            }
        else:
            decision = {"action": "finish", "goal_result": "succeeded", "summary": f"Completed inspection for {compilation}"}
        return call("SupervisorResponse", {"decision": decision}, index)

    model = ScriptedChatModel(responder=respond)
    supervisor = ModelSupervisor(
        tmp_path, settings, instructions=load_prompt("supervisor.md"), model=model
    )

    def worker(state):
        return {
            "result": WorkResult(
                request_id=state["request"]["request_id"], outcome="completed", summary="Inspected"
            ).model_dump()
        }

    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_compilation_graph(
            {"worker": AgentSpec("Inspect", worker, operations=("inspect",))}, checkpointer=saver
        )
        for identifier in ("one", "two"):
            result = graph.invoke(
                {"compilation_id": identifier, "goal": "Inspect"},
                {"configurable": {"thread_id": identifier}},
                context=CompilationContext(tmp_path, supervisor, "Inspect"),
            )
            assert result["status"] == "stopped" and len(result["history"]) == 2
    assert summaries and seen == {"one", "two"}
