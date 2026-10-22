"""Run-owned configuration and prompt snapshots, with no Agent assembly or legacy BIR."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from medbenchagent import prompts
from medbenchagent.settings import (
    CodingSettings, SupervisorSettings, get_coding_api_key, get_llm_api_key,
    load_coding_settings, load_project_env, load_supervisor_settings,
)

from .contracts import Contract, Identifier, Text


class RunOptions(Contract):
    compilation_id: Identifier
    goal: Text
    supervisor_mode: Literal["scripted", "llm"]
    coding_mode: Literal["scripted", "openhands"]
    planner_mode: Literal["scripted"] = "scripted"
    initial_evidence: bool
    pause_for_input: bool
    supervisor_settings: dict[str, Any] | None
    coding_settings: dict[str, Any] | None
    prompts: dict[str, prompts.PromptSnapshot]

    @model_validator(mode="after")
    def check_selected_implementations(self):
        required = set()
        for agent, mode, active, snapshot, schema in (
            ("supervisor", self.supervisor_mode, "llm", self.supervisor_settings, SupervisorSettings),
            ("coding", self.coding_mode, "openhands", self.coding_settings, CodingSettings),
        ):
            if mode == active:
                if snapshot is None:
                    raise ValueError(f"Missing frozen {agent} settings")
                if not schema.from_snapshot(snapshot).llm.model:
                    raise ValueError(f"Missing frozen {agent} model identity")
                required.add(agent)
            elif snapshot is not None:
                raise ValueError(f"Scripted {agent} cannot contain model settings")
        if set(self.prompts) != required:
            raise ValueError("Prompt snapshots must match the selected model-driven agents")
        if self.supervisor_mode == "llm" and self.pause_for_input:
            raise ValueError("--pause-for-input belongs to the scripted Supervisor")
        return self


def create_run(
    directory: Path, *, compilation_id: str, goal: str,
    supervisor_mode: str = "scripted", coding_mode: str = "scripted",
    config_path: Path | None = None, initial_evidence: bool = False, pause_for_input: bool = False,
) -> RunOptions:
    directory = directory.resolve()
    path = directory / "run.json"
    if path.exists() or (directory / "checkpoints.sqlite").exists():
        raise ValueError("This directory already contains a run; use resume")
    if supervisor_mode == coding_mode == "scripted" and config_path is not None:
        raise ValueError("--config requires --supervisor llm or --coding openhands")
    if supervisor_mode == "llm" and pause_for_input:
        raise ValueError("--pause-for-input belongs to the scripted Supervisor")
    snapshots, instructions = {}, {}
    for agent, mode, active, loader, credential in (
        ("supervisor", supervisor_mode, "llm", load_supervisor_settings, get_llm_api_key),
        ("coding", coding_mode, "openhands", load_coding_settings, get_coding_api_key),
    ):
        if mode == active:
            from medbenchagent.runtime.model_services import resolve_model_service

            settings = loader(config_path)
            model = resolve_model_service(agent.capitalize(), settings.llm.base_url,
                                          settings.llm.model, credential())
            snapshots[agent] = replace(settings, llm=replace(settings.llm, model=model)).snapshot()
            instructions[agent] = prompts.freeze_prompt(f"{agent}.md")
    options = RunOptions(
        compilation_id=compilation_id, goal=goal, supervisor_mode=supervisor_mode,
        coding_mode=coding_mode, initial_evidence=initial_evidence, pause_for_input=pause_for_input,
        supervisor_settings=snapshots.get("supervisor"), coding_settings=snapshots.get("coding"),
        prompts=instructions,
    )
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(options.model_dump_json(indent=2) + "\n")
    return options


def resume_run(directory: Path, compilation_id: str) -> RunOptions:
    options = RunOptions.model_validate_json((directory / "run.json").read_text(encoding="utf-8"))
    if options.compilation_id != compilation_id:
        raise ValueError("The directory belongs to another Compilation")
    if not (directory / "checkpoints.sqlite").is_file():
        raise ValueError("No saved checkpoint exists")
    if options.prompts:
        # Credentials may rotate; neither live settings nor prompt resources are read.
        load_project_env()
    return options
