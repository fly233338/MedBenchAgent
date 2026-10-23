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

        if options.supervisor_mode == "llm":
            from medbenchagent.settings import SupervisorSettings
            from .supervisor import ModelSupervisor

            settings = SupervisorSettings.from_snapshot(options.supervisor_settings)
            decide = ModelSupervisor(directory, settings, model=model_for(settings),
                                     instructions=options.prompts["supervisor"].text)
        else:
            from .demo import scripted_supervisor
            decide = scripted_supervisor(directory, pause_for_input=options.pause_for_input)
        if options.coding_mode == "openhands":
            from medbenchagent.settings import CodingSettings
            from .coding import OpenHandsCodingAgent

            coding = OpenHandsCodingAgent(CodingSettings.from_snapshot(options.coding_settings),
                                         instructions=options.prompts["coding"].text).spec()
        else:
            from .demo import scripted_coding_spec
            coding = scripted_coding_spec()
        if options.planner_mode == "llm":
            from medbenchagent.settings import TaskPlannerSettings
            from medbenchagent.agents.semantic_planner import build_planner_spec
            settings = TaskPlannerSettings.from_snapshot(options.planner_settings)
            planner = build_planner_spec(settings, model=model_for(settings), instructions=options.prompts["planner"].text)
        else:
            from .demo import scripted_planner_spec
            planner = scripted_planner_spec()
        agents = {"coding": coding, "task_planner": planner}
        if options.dataset_manifest is not None and options.coding_mode == "scripted":
            # Synthetic Coding is not advertised as a capability on real data.
            agents.pop("coding")
        additional_agents = dict(additional_agents or {})
        if additional_agents.keys() & agents.keys():
            raise ValueError("Additional agents cannot replace a frozen built-in selection")
        agents.update(additional_agents)
        yield agents, CompilationContext(directory.resolve(), decide, options.goal)
