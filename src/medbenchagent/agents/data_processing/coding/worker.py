"""Local SDK subprocess; its only durable execution state belongs to OpenHands.

The host may import candidate types without importing the optional SDK. The worker
accepts a WorkRequest plus resolved paths/settings on stdin and returns one candidate
on stdout. Neither this transport nor candidates are a second agent calling protocol.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Literal
import uuid

from pydantic import Field

from medbenchagent.compilation.contracts import Contract, Identifier, Text, WorkIssue, WorkOutcome, WorkRequest
from .skills import FrozenSkills, load_native_skills
from medbenchagent.runtime.construction import record_event


class CandidateFile(Contract):
    path: Text
    kind: Text


class CandidateNeed(Contract):
    need_id: Identifier
    kind: Literal["work", "user"]
    summary: Text


class CodingCandidate(Contract):
    request_id: Identifier
    outcome: WorkOutcome
    summary: Text
    files: list[CandidateFile] = Field(default_factory=list)
    issues: list[WorkIssue] = Field(default_factory=list)
    needs: list[CandidateNeed] = Field(default_factory=list)


class CodingSubmission(Contract):
    # FinishTool reserves top-level summary/kind/message, hence the envelope.
    result: CodingCandidate


def failure(request_id: str, outcome: WorkOutcome, code: str, message: str) -> CodingCandidate:
    return CodingCandidate(
        request_id=request_id, outcome=outcome, summary=message,
        issues=[WorkIssue(code=code, message=message)],
    )


def _host_message(event) -> bool:
    return (getattr(event, "source", None) == "user"
            and getattr(getattr(event, "llm_message", None), "role", None) == "user"
            and str(getattr(event, "sender", "") or "").startswith("compilation:"))


def _turn_events(conversation, request_id: str, message: str):
    events = list(conversation.state.events)
    matches = [index for index, event in enumerate(events)
               if _host_message(event) and event.sender == f"compilation:{request_id}"]
    if not matches:
        return None, False
    if len(matches) != 1:
        raise ValueError("Duplicate durable Host messages for the same request")
    start = matches[0]
    original = "".join(getattr(part, "text", "") for part in events[start].llm_message.content)
    if original != message:
        raise ValueError("A request ID cannot be reused with different work or input locations")
    end = next((index for index in range(start + 1, len(events)) if _host_message(events[index])),
               len(events))
    return events[start + 1:end], end < len(events)


def execute_turn(conversation, finish_tool, request: WorkRequest, prompt: str) -> CodingCandidate:
    """Replay one request using only SDK-owned messages and typed FinishTool events."""
    message = f"[host-turn:{request.request_id}]\n{prompt}"
    events, superseded = _turn_events(conversation, request.request_id, message)

    def submitted(turn_events):
        raw = finish_tool.parse_last_response(turn_events)
        if raw is None:
            return None
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="json")
        candidate = CodingSubmission.model_validate(raw).result
        if candidate.request_id != request.request_id:
            raise ValueError("FinishTool returned another request's result")
        return candidate

    if events is not None:
        if candidate := submitted(events):
            return candidate
        if superseded:
            return failure(request.request_id, "failed", "superseded_request",
                           "This unfinished request was already followed by another Host request")
    else:
        conversation.send_message(message, sender=f"compilation:{request.request_id}")
    conversation.run()
    events, _ = _turn_events(conversation, request.request_id, message)
    if candidate := submitted(events or []):
        return candidate
    status = getattr(conversation.state.execution_status, "value", conversation.state.execution_status)
    return failure(request.request_id, "blocked" if status == "stuck" else "failed",
                   "sdk_incomplete", f"OpenHands stopped without a submitted result (status={status})")


def activity_callback(payload, secrets=()):
    """Project only executable actions and observations, never SDK reasoning messages."""
    from openhands.sdk.event import ActionEvent, ObservationEvent
    from openhands.sdk.event.conversation_error import ConversationErrorEvent
    from time import perf_counter
    started = {}
    identity = {"compilation_id": payload["request"]["compilation_id"], "agent_id": "coding",
                "request_id": payload["request"]["request_id"], "parent_request_id": payload.get("parent_request_id"),
                "operation": payload.get("parent_operation") or "execute", "workstream_id": payload["request"]["workstream_id"]}
    path = Path(payload["run_directory"]) / "events.jsonl"
    def observe(event):
        values = {**identity, "event_id": payload["conversation_id"] + ":" + str(event.id)}
        if isinstance(event, ActionEvent):
            started[event.tool_call_id] = perf_counter()
            record_event(path, "coding_action", redactions=secrets, **values, tool_name=event.tool_name,
                         tool_call_id=event.tool_call_id, arguments=event.action.model_dump(mode="json") if event.action else {})
        elif isinstance(event, ObservationEvent):
            beginning = started.pop(event.tool_call_id, None)
            output = "".join(getattr(part, "text", "") for part in event.observation.to_llm_content)
            record_event(path, "coding_observation", redactions=secrets, **values, tool_name=event.tool_name,
                         tool_call_id=event.tool_call_id, output=output, status="failed" if event.observation.is_error else "completed",
                         elapsed_ms=int((perf_counter() - beginning) * 1000) if beginning is not None else None)
        elif isinstance(event, ConversationErrorEvent):
            record_event(path, "coding_error", redactions=secrets, **values, summary=event.detail)
    return observe


def _conversation(payload):
    runtime = payload["settings"]["coding"]
    for package, expected in runtime["sdk_packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Frozen Coding dependency differs: {package} requires {expected}")

    from pydantic import SecretStr
    from openhands.sdk import Agent, Conversation, LLM, Tool
    from openhands.sdk.context import AgentContext
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.tool.registry import resolve_tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    # Only the in-memory LLM receives the key; terminal children do not inherit it.
    key = os.environ.pop("MEDBENCH_CODING_LLM_API_KEY", None)
    os.environ.pop("MEDBENCH_LLM_API_KEY", None)
    if not key:
        raise RuntimeError("Coding credential was not supplied by the Host")
    model = payload["settings"]["llm"]
    name = model["model"]
    llm = LLM(
        # /models supplied an exact served ID, which may itself contain slashes.
        model=f"openai/{name}",
        base_url=model["base_url"], api_key=SecretStr(key),
        usage_id="compilation_coding", num_retries=model["max_retries"],
        timeout=max(1, math.ceil(model["timeout_seconds"])),
        temperature=model["default_temperature"],
        reasoning_effort=model["reasoning_effort"] or None,
        max_output_tokens=runtime["max_output_tokens"],
    )
    finish_spec = Tool(name="FinishTool", params={"response_schema": CodingSubmission.model_json_schema()})
    agent = Agent(
        llm=llm,
        agent_context=AgentContext(
            skills=load_native_skills(Path(payload["run_directory"]), FrozenSkills.model_validate(payload["skills"])),
            load_user_skills=False, load_public_skills=False, load_project_skills=False, load_memory=False,
        ),
        tools=[Tool(name=TerminalTool.name, params={"terminal_type": "powershell"} if os.name == "nt" else {}),
               Tool(name=FileEditorTool.name), finish_spec],
        include_default_tools=["ThinkTool"],
        condenser=LLMSummarizingCondenser(
            llm=llm.model_copy(update={"usage_id": "compilation_coding_context"}),
            max_size=runtime["condenser_max_events"], keep_first=2,
            max_tokens=runtime["condenser_max_tokens"],
        ),
    )
    conversation = Conversation(
        agent=agent, workspace=Path(payload["workspace"]),
        persistence_dir=Path(payload["persistence_dir"]),
        conversation_id=uuid.UUID(payload["conversation_id"]),
        max_iteration_per_run=runtime["max_iterations"],
        stuck_detection=True, delete_on_close=False, visualizer=None,
        observability_metadata={name: payload["request"][name]
                                for name in ("compilation_id", "request_id", "agent_id", "workstream_id")},
        callbacks=[activity_callback(payload, [key])],
    )
    # Resolving the builtin parser does not initialize terminal tools or call a model.
    return conversation, resolve_tool(finish_spec, conversation.state)[0]


def main():
    payload = json.load(sys.stdin)
    request = WorkRequest.model_validate(payload["request"])
    conversation = None
    secrets = [os.environ.get(name) for name in ("MEDBENCH_CODING_LLM_API_KEY", "MEDBENCH_LLM_API_KEY")]

    def diagnostic(exc):
        message = f"{type(exc).__name__}: {exc}"
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[redacted]")
        return message[:2000]

    # SDK/tool diagnostics belong to stderr; stdout is solely the candidate transport.
    with redirect_stdout(sys.stderr):
        try:
            conversation, finish_tool = _conversation(payload)
        except Exception as exc:
            candidate = failure(request.request_id, "blocked", "environment_unavailable",
                                f"Cannot open the frozen Coding environment: {diagnostic(exc)}")
            print(f"Coding setup: {diagnostic(exc)}", file=sys.stderr)
        else:
            try:
                candidate = execute_turn(conversation, finish_tool, request, payload["prompt"])
            except Exception as exc:
                candidate = failure(request.request_id, "failed", "coding_execution_failed",
                                    f"OpenHands execution failed: {diagnostic(exc)}")
                print(f"Coding execution: {diagnostic(exc)}", file=sys.stderr)
            finally:
                conversation.close()
    print(candidate.model_dump_json())


if __name__ == "__main__":
    main()
