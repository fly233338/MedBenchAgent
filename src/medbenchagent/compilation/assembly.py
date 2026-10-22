"""Explicit composition of selected implementations; no data creation or plugin manager."""

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

from .interfaces import AgentSpec, CompilationContext
from .run import RunOptions


@contextmanager
def assemble_compilation(
    options: RunOptions, directory: Path, *, additional_agents: Mapping[str, AgentSpec] | None = None,
):
    """Yield the registry and injected context, then release owned model clients."""
    cleanup = None
    try:
        if options.supervisor_mode == "llm":
            from medbenchagent.settings import SupervisorSettings
            from medbenchagent.runtime.llm import clear_client_cache
            from .supervisor import ModelSupervisor

            cleanup = clear_client_cache
            decide = ModelSupervisor(directory, SupervisorSettings.from_snapshot(options.supervisor_settings),
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
        # The current run contract explicitly selects a scripted Planner.
        from .demo import scripted_planner_spec
        agents = {"coding": coding, "task_planner": scripted_planner_spec()}
        additional_agents = dict(additional_agents or {})
        if additional_agents.keys() & agents.keys():
            raise ValueError("Additional agents cannot replace a frozen built-in selection")
        agents.update(additional_agents)
        yield agents, CompilationContext(directory.resolve(), decide, options.goal)
    finally:
        if cleanup is not None:
            cleanup()
