"""Explicit composition of selected implementations; no data creation or plugin manager."""

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .interfaces import AgentSpec, CompilationContext
from .run import RunOptions


@contextmanager
def assemble_compilation(
    options: RunOptions, directory: Path, *, additional_agents: Mapping[str, AgentSpec] | None = None,
):
    """Yield the registry and injected context, then release owned model clients."""
    with ExitStack() as stack:
        def model_for(settings):
            from medbenchagent.runtime.native import chat_model
            model = chat_model(settings.llm, settings.loop)
            if getattr(model, "root_client", None) is not None:
                stack.callback(model.root_client.close)
            return model

        from medbenchagent.settings import SupervisorSettings
        from medbenchagent.agents.supervisor.agent import ModelSupervisor

        settings = SupervisorSettings.from_snapshot(options.supervisor_settings)
        decide = ModelSupervisor(directory, settings, model=model_for(settings),
                                 instructions=options.prompts["supervisor"].text)
        from medbenchagent.settings import CodingSettings
        from medbenchagent.agents.data_processing.coding.adapter import OpenHandsCodingAgent

        coding_agent = OpenHandsCodingAgent(CodingSettings.from_snapshot(options.coding_settings),
                                           instructions=options.prompts["coding"].text, skills=options.coding_skills)
        from medbenchagent.settings import TaskPlannerSettings
        from medbenchagent.agents.task_planning.agent import build_planner_spec
        settings = TaskPlannerSettings.from_snapshot(options.planner_settings)
        planner = build_planner_spec(settings, model=model_for(settings), instructions={key: options.prompts[key].text for key in ("discover_scenarios", "design_leaves")})
        agents = {"task_planner": planner}
        from medbenchagent.settings import WorkAgentSettings
        if options.data_processing_mode == "llm":
            from medbenchagent.agents.data_processing.agent import build_data_processing_spec
            settings = WorkAgentSettings.from_snapshot(options.professional_settings["data_processing"])
            agents["data_processing"] = build_data_processing_spec(settings, coding_agent,
                instructions={key: options.prompts[key].text for key in ("profile", "prepare_task_data", "coding_handoff")}, model=model_for(settings))
        if options.generation_mode == "llm":
            from medbenchagent.agents.generation.agent import build_generation_spec
            settings = WorkAgentSettings.from_snapshot(options.professional_settings["generation"])
            agents["generation"] = build_generation_spec(settings, instructions={"generate": options.prompts["generate"].text},
                                                        model=model_for(settings))
        if options.review_mode == "llm":
            from medbenchagent.agents.review.agent import build_review_spec
            settings = WorkAgentSettings.from_snapshot(options.professional_settings["review"])
            agents["review"] = build_review_spec(settings, instructions={key: options.prompts[key].text for key in ("review_plan", "review_candidate")}, model=model_for(settings))
        if options.delivery_enabled:
            from medbenchagent.agents.delivery.node import delivery_spec
            agents["delivery"] = delivery_spec()
        from functools import partial
        from .policy import attach_checks, check_goal, readiness
        agents = {role: attach_checks(spec, directory, role) for role, spec in agents.items()}
        additional_agents = dict(additional_agents or {})
        if additional_agents.keys() & agents.keys():
            raise ValueError("Additional agents cannot replace a frozen built-in selection")
        agents.update(additional_agents)
        yield agents, CompilationContext(directory.resolve(), decide, options.goal,
            goal_spec=options.goal_spec, repair_limits=options.repair_limits,
            check_goal=partial(check_goal, directory, options.goal_spec, options.benchmark_request),
            readiness=partial(readiness, directory, agents))
