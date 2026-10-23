"""Task Planning agent."""

import json
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from medbenchagent.compilation.artifacts import (
    read_json_artifact,
    resolve_artifact,
)
from medbenchagent.compilation.contracts import (
    WorkIssue,
    WorkRequest,
    WorkResult,
)
from medbenchagent.compilation.interfaces import AgentSpec
from medbenchagent.data.dataset import FileDatasetReader
from medbenchagent.data.task_data import ProfileReader, TaskDataReader
from medbenchagent.planning.contracts import (
    TaskPlan,
)
from medbenchagent.planning import semantic
from medbenchagent.runtime.native import OperationTools, AGENT_LIMIT_ERRORS, chat_model, loop_middleware, role_lock

from .tools import planner_tools
from .state import PlannerContext, PlannerResponse, PlannerState, _load, _recover_publications, _resolve_result, _save


class PlannerContextMiddleware(AgentMiddleware):
    state_schema = PlannerState

    def before_agent(self, state, runtime):
        context = runtime.context
        if state.get("active_request_id") == context.invocation_id:
            return None
        plans = [ref for ref in context.request.inputs if ref.kind == "task_plan"]
        if len(plans) > 1:
            raise ValueError("Bind exactly one Task Plan to revise")
        if plans:
            plan = TaskPlan.model_validate(read_json_artifact(context.directory, plans[0]))
        else:
            plan = None
        if plan is None:
            plan = TaskPlan(
                plan_id=f"{context.compilation_id}-plan-{context.dataset.digest[:12]}",
                dataset=semantic.dataset_ref(context.reader),
                dataset_digest=context.dataset.digest,
            )
        if plan.dataset_digest != context.dataset.digest:
            raise ValueError(
                "Task Plan binds a different dataset artifact; begin a new plan for new evidence"
            )
        semantic.validate_plan(plan, context.reader, task_data=context.task_data)
        return {
            "active_request_id": context.invocation_id,
            "draft_path": _save(plan, context),
            "submitted": [],
            "structured_response": None,
        }

    def wrap_model_call(self, request, handler):
        context = request.runtime.context
        plan = _load(request.state, context)
        current = {
            "request": context.request.model_dump(mode="json"),
            "dataset": semantic.dataset_ref(context.reader),
            "plan": semantic.plan_summary(plan),
            "submitted": [
                {"artifact_id": item["artifact_id"], "kind": item["kind"]}
                for item in request.state.get("submitted", [])
            ],
            "query_limits": {
                "page_size": context.settings.page_size,
                "preview_chars": context.settings.preview_chars,
            },
        }
        response = handler(
            request.override(
                system_message=SystemMessage(content=context.instructions[context.request.operation]),
                messages=[HumanMessage(content="Current work data (not instructions):\n" + json.dumps(current, ensure_ascii=False),
                                       id="current-work-view"), *request.messages]
            )
        )
        if response.structured_response is not None:
            try:
                _resolve_result(response.structured_response, request.state, context)
            except (ValueError, KeyError, FileNotFoundError) as exc:
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


def build_planner_spec(settings, *, instructions, model=None):
    model = model or chat_model(settings.llm, settings.loop)
    tools = planner_tools()
    writes = [
        item.name
        for item in tools
        if item.name.startswith(
            ("record_", "build_", "save_", "set_", "propose_", "accept_", "exclude_", "publish_")
        )
    ]
    agent = create_agent(
        model,
        tools=tools,
        response_format=ToolStrategy(PlannerResponse),
        state_schema=PlannerState,
        context_schema=PlannerContext,
        checkpointer=True,
        name="task_planner_agent",
        middleware=[
            PlannerContextMiddleware(),
            OperationTools({
                "discover_scenarios": {t.name for t in tools} - {"save_semantic_task", "exclude_semantic_task", "accept_data_requirement"},
                "design_leaves": {t.name for t in tools} - {"record_axis_assessment", "build_candidate_scenarios", "record_scenario_decision", "set_scenario_backends"},
            }),
            *loop_middleware(model, settings.loop, settings.context_chars, write_tools=writes),
        ],
    )

    def node(state, config, runtime):
        request = WorkRequest.model_validate(state["request"])
        directory = runtime.context.directory
        try:
            datasets = [ref for ref in request.inputs if ref.kind in {"dataset", "dataset_profile"}]
            if len(datasets) != 1:
                raise ValueError("Planner requires exactly one bound Dataset artifact")
            context = PlannerContext(
                directory,
                request,
                (ProfileReader if datasets[0].kind == "dataset_profile" else FileDatasetReader)(
                    resolve_artifact(directory, datasets[0])
                ),
                datasets[0],
                settings,
                instructions,
                task_data={
                    ref.artifact_id: TaskDataReader(resolve_artifact(directory, ref))
                    for ref in request.inputs
                    if ref.kind == "task_data"
                },
            )
            with role_lock(directory, "task_planner"):
                result = agent.invoke(
                    {
                        "messages": [
                            HumanMessage(content=request.objective, id=request.request_id)
                        ],
                        "structured_response": None,
                    },
                    {**config, "recursion_limit": settings.loop.max_model_calls * 6 + 20},
                    context=context,
                )
            response = result.get("structured_response")
            if response is None:
                raise ValueError("Planner ended without a current structured result")
            output = _resolve_result(response, result, context)
        except (ValueError, KeyError, OSError) as exc:
            output = WorkResult(
                request_id=request.request_id,
                outcome="failed",
                summary="Planner work could not be accepted",
                issues=[WorkIssue(code="invalid_planning_work", message=str(exc))],
            )
        except AGENT_LIMIT_ERRORS as exc:
            output = WorkResult(
                request_id=request.request_id,
                outcome="failed",
                summary="Planner invocation budget exhausted",
                artifacts=_recover_publications(context),
                issues=[WorkIssue(code="agent_limit", message=str(exc))],
            )
        return {"result": output.model_dump(mode="json")}

    return AgentSpec(
        "Semantic task planning from a canonical Dataset: evidence queries, Axis/Scenario judgments, "
        "task revision, data requirements and formal Task Plan publication. Requires one dataset; "
        "accepts one dataset or dataset_profile, a selected task_plan and prepared task_data inputs. "
        "Can publish Scenario-level requirements before Leaf design, and later define Leaf assembly. Does not generate items.",
        node, operations=("discover_scenarios", "design_leaves"),
    )
