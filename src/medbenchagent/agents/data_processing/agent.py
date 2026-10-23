"""Data Processing agent."""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.runtime.native import OperationTools, chat_model, loop_middleware
from medbenchagent.agents._shared.handoff import (
    WorkResponse,
    WorkInstructions,
    invoke_work,
)

from .tools import data_processing_tools
from .state import DataProcessingContext, DataProcessingState


def build_data_processing_spec(settings, coding, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    agent = create_agent(
        model,
        tools=data_processing_tools(),
        response_format=ToolStrategy(WorkResponse),
        state_schema=DataProcessingState,
        context_schema=DataProcessingContext,
        checkpointer=True,
        name="data_processing_agent",
        middleware=[
            WorkInstructions(DataProcessingState),
            OperationTools({'profile': None, 'prepare_task_data': None}),
            *loop_middleware(
                model,
                settings.loop,
                settings.context_chars,
                write_tools=(
                    "invoke_coding",
                    "validate_data_candidate",
                    "inspect_candidate_cases",
                    "submit_data_artifacts",
                ),
            ),
        ],
    )

    def node(state, config, runtime):
        context = DataProcessingContext(
            runtime.context.directory,
            WorkRequest.model_validate(state["request"]),
            settings,
            instructions,
            "data_processing",
            coding,
            runtime.context.goal,
        )
        return invoke_work(agent, context, config)

    return AgentSpec(
        "Inspect delegated raw sources, produce evidence-grounded Profiles, prepare backend-scoped TaskData, "
        "and validate/submit data. Owns internal OpenHands Coding calls and local repair; requires source_manifest, "
        "dataset_profile or dataset inputs and any applicable TaskPlan/data requirements.",
        node, operations=('profile', 'prepare_task_data'),
    )
