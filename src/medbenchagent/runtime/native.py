"""Small LangChain integration helpers, not another agent runtime or state store."""

import json
from pathlib import Path

from filelock import FileLock
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langgraph.errors import GraphRecursionError

from medbenchagent.runtime.construction import record_event
from medbenchagent.runtime.llm import RequestContextBudgetError
from medbenchagent.settings import get_llm_api_key


AGENT_LIMIT_ERRORS = (ModelCallLimitExceededError, ToolCallLimitExceededError, GraphRecursionError)


class OperationTools(AgentMiddleware):
    """Expose only the current operation's tools and reject stale tool calls."""

    def __init__(self, operations):
        self.operations = operations

    def allowed(self, context, name):
        allowed = self.operations[context.request.operation]
        return allowed is None or name in allowed

    def wrap_model_call(self, request, handler):
        return handler(request.override(tools=[t for t in request.tools
            if self.allowed(request.runtime.context, t.name)]))

    def wrap_tool_call(self, request, handler):
        if not self.allowed(request.runtime.context, request.tool_call["name"]):
            return ToolMessage(content="Tool is not available for the current operation",
                               tool_call_id=request.tool_call["id"], status="error")
        return handler(request)


def chat_model(settings, limits):
    """Use the configured Chat-compatible transport without model discovery."""
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=get_llm_api_key(),
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        temperature=settings.default_temperature,
        max_tokens=limits.max_output_tokens,
        reasoning_effort=settings.reasoning_effort or None,
        use_responses_api=False,
    )


def role_lock(directory: Path, role: str):
    directory = directory / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    return FileLock(directory / f"{role}.lock", timeout=0)


class InvocationGuard(AgentMiddleware):
    """Bound actual requests and make expected tool errors actionable observations."""

    def __init__(self, limits, context_chars, *, write_tools=()):
        self.limits = limits
        self.context_chars = context_chars
        self.write_tools = frozenset(write_tools)

    def wrap_model_call(self, request, handler):
        messages = ([request.system_message] if request.system_message else []) + request.messages
        tools = [convert_to_openai_tool(item) for item in request.tools]
        schema = getattr(request.response_format, "schema", None)
        schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else str(schema)
        definitions = json.dumps({"tools": tools, "response": schema}, ensure_ascii=False)
        serialized = json.dumps(
            [item.model_dump(mode="json") for item in messages], ensure_ascii=False
        )
        tokens = (
            count_tokens_approximately(messages)
            + len(definitions) // 3
            + self.limits.max_output_tokens
        )
        if (
            tokens > self.limits.context_tokens
            or len(serialized) + len(definitions) + self.limits.max_output_tokens * 4
            > self.context_chars
        ):
            raise RequestContextBudgetError(
                len(serialized) + len(definitions) + self.limits.max_output_tokens * 4,
                self.context_chars,
            )
        context = request.runtime.context
        record_event(
            context.directory / "events.jsonl",
            "agent_model_call",
            compilation_id=context.compilation_id,
            agent_id=context.agent_id,
            request_id=context.invocation_id,
            estimated_tokens=tokens,
        )
        # This provider option is also enforced for writes on the code side below.
        return handler(
            request.override(
                model_settings={**request.model_settings, "parallel_tool_calls": False}
            )
        )

    def wrap_tool_call(self, request, handler):
        calls = next(
            (
                message.tool_calls
                for message in reversed(request.state["messages"])
                if getattr(message, "tool_calls", None)
            ),
            [],
        )
        try:
            if (
                request.tool_call["name"] in self.write_tools
                and sum(call["name"] in self.write_tools for call in calls) > 1
            ):
                raise ValueError(
                    "Call only one state-changing tool per model turn; then read its new revision"
                )
            result = handler(request)
            context = request.runtime.context
            record_event(
                context.directory / "events.jsonl",
                "agent_tool_result",
                compilation_id=context.compilation_id,
                agent_id=context.agent_id,
                request_id=context.invocation_id,
                tool_name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="completed",
            )
            return result
        except (ValueError, KeyError, FileNotFoundError) as exc:
            context = request.runtime.context
            record_event(
                context.directory / "events.jsonl",
                "agent_tool_result",
                compilation_id=context.compilation_id,
                agent_id=context.agent_id,
                request_id=context.invocation_id,
                tool_name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="rejected",
                error_type=type(exc).__name__,
            )
            return ToolMessage(
                content=json.dumps({"error": type(exc).__name__, "detail": str(exc)}),
                tool_call_id=request.tool_call["id"],
                status="error",
            )


def loop_middleware(model, limits, context_chars, *, write_tools=()):
    return [
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", limits.summary_trigger_tokens),
            keep=("tokens", limits.summary_keep_tokens),
        ),
        ModelCallLimitMiddleware(run_limit=limits.max_model_calls, exit_behavior="error"),
        ToolCallLimitMiddleware(run_limit=limits.max_tool_calls, exit_behavior="error"),
        InvocationGuard(limits, context_chars, write_tools=write_tools),
    ]
