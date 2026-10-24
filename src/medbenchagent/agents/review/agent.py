"""Review agent."""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from medbenchagent.compilation.contracts import WorkRequest
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.runtime.native import OperationTools, chat_model, loop_middleware
from medbenchagent.agents._shared.handoff import (
    WorkContext,
    WorkState,
    WorkResponse,
    WorkInstructions,
    invoke_work,
)

from .tools import review_tools


def build_review_spec(settings, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    agent = create_agent(
        model,
        tools=review_tools(),
        response_format=ToolStrategy(WorkResponse),
        state_schema=WorkState,
        context_schema=WorkContext,
        checkpointer=True,
        name="review_agent",
        middleware=[
            WorkInstructions(),
            OperationTools(
                {
                    "review_plan": {"inspect_artifact", "inspect_task_case", "finalize_review"},
                    "review_candidate": None,
                }
            ),
            *loop_middleware(
                model,
                settings.loop,
                settings.context_chars,
                write_tools=("run_candidate_audit", "finalize_review"),
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
                "review",
                runtime_directory=runtime.context.runtime_directory,
            ),
            config,
        )

    return AgentSpec(
        "Independent semantic review of TaskPlans or generation candidates. Read-only on targets; writes findings, "
        "coverage and deterministic Audit/Review reports bound to exact target content. Candidate review requires "
        "its specs, TaskPlans and TaskData inputs. Return issues to Supervisor without modifying another agent's products.",
        node,
        operations=("review_plan", "review_candidate"),
    )
