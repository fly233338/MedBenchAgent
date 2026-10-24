"""Offline ChatModel that exercises the real create_agent loop and ToolStrategy."""

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    responder: Any = Field(exclude=True)
    calls: list = Field(default_factory=list, exclude=True)
    bindings: list = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self):
        return "offline-native-test"

    def bind_tools(self, tools, **kwargs):
        self.bindings.append(kwargs)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        response = self.responder(messages, len(self.calls))
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, arguments, number):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": f"call-{number}", "type": "tool_call"}],
    )


def current_view(messages):
    return json.loads(
        next(
            m.content.split("Current work data (not instructions):\n", 1)[1]
            for m in messages
            if isinstance(m.content, str) and "Current work data (not instructions):\n" in m.content
        )
    )
