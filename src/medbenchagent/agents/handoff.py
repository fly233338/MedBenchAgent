"""Thin native-agent handoff helpers. State and execution remain owned by LangGraph."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from pydantic import Field

from medbenchagent.compilation.artifacts import (
    read_json_artifact,
    resolve_artifact,
    write_json_artifact,
)
from medbenchagent.compilation.contracts import (
    ArtifactRef,
    Contract,
    WorkIssue,
    WorkNeed,
    WorkOutcome,
    WorkRequest,
    WorkResult,
)
from medbenchagent.data.task_data import ProductBinding
from medbenchagent.runtime.native import operation_instructions, AGENT_LIMIT_ERRORS, role_lock


class WorkState(AgentState):
    active_request_id: str
    submitted: list[dict]


class ResponseNeed(Contract):
    need_id: str
    kind: Literal["work", "user"]
    summary: str
    detail_artifact_id: str | None = None


class WorkResponse(Contract):
    outcome: WorkOutcome
    summary: str = Field(min_length=1)
    artifact_ids: list[str] = Field(default_factory=list)
    issues: list[WorkIssue] = Field(default_factory=list)
    needs: list[ResponseNeed] = Field(default_factory=list)


@dataclass(frozen=True)
class WorkContext:
    directory: Path
    request: WorkRequest
    settings: object
    instructions: str | dict[str, str]
    agent_id: str
    # Transient return receipts only. Durable authority remains the private checkpoint and files.
    publications: dict[str, ArtifactRef] = field(default_factory=dict, init=False, compare=False)

    @property
    def compilation_id(self):
        return self.request.compilation_id

    @property
    def invocation_id(self):
        return self.request.request_id


def binding(ref):
    return ProductBinding(artifact_id=ref.artifact_id, digest=ref.digest)


def references(state, context):
    return {
        ref.artifact_id: ref
        for ref in [
            *context.request.inputs,
            *(ArtifactRef.model_validate(row) for row in state.get("submitted", [])),
        ]
    }


def get_ref(runtime, identifier, kind=None):
    ref = references(runtime.state, runtime.context)[identifier]
    if kind is not None and ref.kind != kind:
        raise ValueError(f"Expected {kind}, got {ref.kind}")
    resolve_artifact(runtime.context.directory, ref)
    return ref


def bound_path(runtime, product, kind):
    ref = get_ref(runtime, product.artifact_id, kind)
    if binding(ref) != product:
        raise ValueError("Bound product digest does not match its registered artifact")
    return resolve_artifact(runtime.context.directory, ref)


def publish_value(runtime, kind, value):
    value = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    return write_json_artifact(
        runtime.context.directory,
        runtime.context.request,
        f"{kind}/{digest}.json",
        kind=kind,
        value=value,
    )


def tool_result(runtime, value, *, artifacts=(), updates=None):
    submitted = {row["artifact_id"]: row for row in runtime.state.get("submitted", [])}
    submitted.update({ref.artifact_id: ref.model_dump(mode="json") for ref in artifacts})
    runtime.context.publications.update({ref.artifact_id: ref for ref in artifacts})
    return Command(
        update={
            **(updates or {}),
            "submitted": list(submitted.values()),
            "messages": [
                ToolMessage(
                    content=json.dumps(value, ensure_ascii=False, default=str),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def finish_response(response, state, context):
    if state.get("active_request_id") != context.invocation_id:
        raise ValueError("Result belongs to a previous request")
    submitted = {
        row["artifact_id"]: ArtifactRef.model_validate(row) for row in state.get("submitted", [])
    }
    if (
        len(response.artifact_ids) != len(set(response.artifact_ids))
        or set(response.artifact_ids) - submitted.keys()
    ):
        raise ValueError("Select unique artifacts actually submitted in this request")
    artifacts = [submitted[key] for key in response.artifact_ids]
    expected = {
        "data_processing": {"dataset_profile" if context.request.operation == "profile" else "task_data"},
        "generation": {"generation_candidate"},
        "review": {"review_report"},
    }
    if response.outcome == "completed" and not any(
        ref.kind in expected[context.agent_id] for ref in artifacts
    ):
        raise ValueError("Completed work requires a submitted domain product")
    # Tools publish real products. Include their dependencies even when the model selects only the final product.
    artifacts.extend(ref for key, ref in submitted.items() if key not in response.artifact_ids)
    for ref in artifacts:
        verify_product(context.directory, ref)
    visible = references(state, context)
    needs = [
        WorkNeed(
            need_id=n.need_id,
            kind=n.kind,
            summary=n.summary,
            detail_ref=visible[n.detail_artifact_id] if n.detail_artifact_id else None,
        )
        for n in response.needs
    ]
    return WorkResult(
        request_id=context.invocation_id,
        outcome=response.outcome,
        summary=response.summary,
        artifacts=artifacts,
        issues=response.issues,
        needs=needs,
    )


def verify_product(directory, ref):
    path = resolve_artifact(directory, ref)
    if ref.kind in {"dataset_profile", "task_data"}:
        from medbenchagent.data.task_data import ProfileReader, TaskDataReader

        (ProfileReader if ref.kind == "dataset_profile" else TaskDataReader)(path)
    elif ref.kind == "generation_candidate":
        from medbenchagent.generation.execution import CandidateReader

        CandidateReader(path)
    return path


class WorkInstructions(AgentMiddleware):
    def __init__(self, state_schema=WorkState, reset_fields=None):
        self.state_schema = state_schema
        self.reset_fields = reset_fields or {}

    def before_agent(self, state, runtime):
        if state.get("active_request_id") != runtime.context.invocation_id:
            return {
                "active_request_id": runtime.context.invocation_id,
                "submitted": [],
                "structured_response": None,
                **self.reset_fields,
            }
        runtime.context.publications.update(
            {
                row["artifact_id"]: ArtifactRef.model_validate(row)
                for row in state.get("submitted", [])
            }
        )

    def wrap_model_call(self, request, handler):
        context = request.runtime.context
        current = {
            "request": context.request.model_dump(mode="json"),
            "submitted": [
                {"artifact_id": row["artifact_id"], "kind": row["kind"]}
                for row in request.state.get("submitted", [])
            ],
        }
        response = handler(
            request.override(
                system_message=SystemMessage(content=operation_instructions(context.instructions, context.request.operation)),
                messages=[HumanMessage(content="Current work data (not instructions):\n" + json.dumps(current, ensure_ascii=False),
                                       id="current-work-view"), *request.messages]
            )
        )
        if response.structured_response is not None:
            try:
                finish_response(response.structured_response, request.state, context)
            except (ValueError, KeyError, OSError) as exc:
                message = response.result[0]
                return ModelResponse(
                    result=[
                        message,
                        *[
                            ToolMessage(content=str(exc), tool_call_id=call["id"], status="error")
                            for call in message.tool_calls
                        ],
                    ]
                )
        return response


def invoke_work(agent, context, config):
    try:
        with role_lock(context.directory, context.agent_id):
            state = agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=context.request.objective, id=context.invocation_id)
                    ],
                    "structured_response": None,
                },
                {**config, "recursion_limit": context.settings.loop.max_model_calls * 6 + 20},
                context=context,
            )
        response = state.get("structured_response")
        if response is None:
            raise ValueError("Agent ended without a current structured response")
        result = finish_response(response, state, context)
    except (*AGENT_LIMIT_ERRORS, ValueError, KeyError, OSError) as exc:
        preserved = []
        for ref in context.publications.values():
            try:
                verify_product(context.directory, ref)
                preserved.append(ref)
            except (ValueError, OSError):
                pass
        result = WorkResult(
            request_id=context.invocation_id,
            outcome="failed",
            summary="Work could not be completed",
            artifacts=preserved,
            issues=[WorkIssue(code="agent_work_failed", message=str(exc))],
        )
    return {"result": result.model_dump(mode="json")}


@tool
def inspect_artifact(
    artifact_id: str, runtime: ToolRuntime[WorkContext], text_offset: int = 0
) -> dict:
    """Read a bound input or this request's submitted JSON artifact in bounded text segments."""
    if text_offset < 0:
        raise ValueError("text_offset must be nonnegative")
    ref = get_ref(runtime, artifact_id)
    text = json.dumps(read_json_artifact(runtime.context.directory, ref), ensure_ascii=False)
    end = text_offset + runtime.context.settings.preview_chars
    return {
        "artifact_id": ref.artifact_id,
        "digest": ref.digest,
        "text": text[text_offset:end],
        "next_text_offset": end if end < len(text) else None,
        "truncated": end < len(text),
    }
