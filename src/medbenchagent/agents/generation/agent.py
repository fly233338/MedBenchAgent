"""Generation agent."""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.runtime.native import OperationTools, chat_model, loop_middleware
from medbenchagent.agents._shared.handoff import (
    WorkContext,
    WorkResponse,
    WorkInstructions,
    invoke_work,
)

from .tools import GenerationState, generation_tools


def build_generation_spec(settings, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    tools = generation_tools()
    agent = create_agent(
        model,
        tools=tools,
        response_format=ToolStrategy(WorkResponse),
        state_schema=GenerationState,
        context_schema=WorkContext,
        checkpointer=True,
        name="generation_agent",
        middleware=[
            WorkInstructions(GenerationState, {"quarantined": {}}),
            OperationTools({"generate": None}),
            *loop_middleware(
                model,
                settings.loop,
                settings.context_chars,
                write_tools=[t.name for t in tools if not t.name.startswith(("inspect_", "list_"))],
            ),
        ],
    )

    def node(state, config, runtime):
        return invoke_work(
            agent,
            WorkContext(
                runtime.context.directory,
                WorkRequest.model_validate(state["request"]),
                settings,
                instructions,
                "generation",
                runtime_directory=runtime.context.runtime_directory,
            ),
            config,
        )

    return AgentSpec(
        "Prepare execution, allocate counts and assemble questions from ready Leaf definitions and accepted TaskData. "
        "Supports VQA, reports, grounding and detection; returns immutable batches/candidates and data or task issues. "
        "Bind TaskPlans, TaskData and any referenced specs/batches explicitly.",
        node,
        operations=("generate",),
    )
