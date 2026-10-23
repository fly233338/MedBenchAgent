"""Run-owned configuration and prompt snapshots, with no Agent assembly or legacy BIR."""

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator, Field

from medbenchagent.agents._shared import prompting as prompts
from medbenchagent.settings import (
    CodingSettings, SupervisorSettings, TaskPlannerSettings, get_coding_api_key, get_llm_api_key,
    load_coding_settings, load_project_env, load_supervisor_settings, load_planner_settings,
    WorkAgentSettings, load_work_agent_settings, load_compilation_target,
)

from .contracts import Contract, Identifier, Text, GoalSpec, RepairLimits


class RunOptions(Contract):
    compilation_id: Identifier
    goal: Text
    goal_spec: GoalSpec = Field(default_factory=GoalSpec)
    repair_limits: RepairLimits = Field(default_factory=RepairLimits)
    implementation_digest: str
    dataset_manifest: str | None = None
    source_manifest: str | None = None
    data_processing_mode: Literal["disabled", "llm"] = "disabled"
    generation_mode: Literal["disabled", "llm"] = "disabled"
    review_mode: Literal["disabled", "llm"] = "disabled"
    delivery_enabled: bool = False
    professional_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    benchmark_request: dict[str, Any] | None = None
    planner_settings: dict[str, Any]
    supervisor_settings: dict[str, Any]
    coding_settings: dict[str, Any]
    prompts: dict[str, prompts.PromptSnapshot]

    @model_validator(mode="after")
    def check_selected_implementations(self):
        required = set()
        enabled = {role for role in ("data_processing", "generation", "review") if getattr(self, f"{role}_mode") == "llm"}
        if set(self.professional_settings) != enabled:
            raise ValueError("Professional settings must match enabled roles")
        for role in enabled:
            if not WorkAgentSettings.from_snapshot(self.professional_settings[role]).llm.model:
                raise ValueError("Enabled Agent requires a frozen model")
            required.update({"data_processing": {"profile", "prepare_task_data", "coding_handoff"}, "generation": {"generate"}, "review": {"review_plan", "review_candidate"}}[role])
        for agent, snapshot, schema in (
            ("supervisor", self.supervisor_settings, SupervisorSettings),
            ("coding", self.coding_settings, CodingSettings),
            ("planner", self.planner_settings, TaskPlannerSettings),
        ):
            if not schema.from_snapshot(snapshot).llm.model:
                raise ValueError(f"Missing frozen {agent} model identity")
            required.update({"planner": {"discover_scenarios", "design_leaves"}}.get(agent, {agent}))
        required.add("structured_repair")
        if set(self.prompts) != required:
            raise ValueError("Prompt snapshots must match the selected model-driven agents")
        return self


def create_run(
    directory: Path, *, compilation_id: str, goal: str,
    config_path: Path | None = None, goal_spec: GoalSpec | None = None, repair_limits: RepairLimits | None = None,
    dataset_manifest: Path | None = None,
    source_manifest: Path | None = None, data_processing_mode: str = "llm",
    generation_mode: str = "llm", review_mode: str = "llm", delivery_enabled: bool = True,
    sample_count: int | None = None,
) -> RunOptions:
    directory = directory.resolve()
    path = directory / "run.json"
    if path.exists() or (directory / "checkpoints.sqlite").exists():
        raise ValueError("This directory already contains a run; use resume")
    if sample_count is not None and (type(sample_count) is not int or sample_count <= 0 or generation_mode != "llm"):
        raise ValueError("sample_count requires enabled Generation and a positive integer")
    if source_manifest is not None and dataset_manifest is not None:
        raise ValueError("Select a source manifest or a canonical Dataset, not both")
    if source_manifest is not None:
        from medbenchagent.data.task_data import SourceManifest
        source = SourceManifest.model_validate_json(source_manifest.read_bytes())
        if any(not (source_manifest.parent / item.root).resolve().is_dir() for item in source.sources):
            raise ValueError("Raw source directories must exist")
        if data_processing_mode != "llm":
            raise ValueError("Raw sources require the Data Processing agent")
    if dataset_manifest is not None:
        from medbenchagent.data.dataset import FileDatasetReader
        FileDatasetReader(dataset_manifest)
    snapshots, instructions = {}, {}
    for agent, loader, credential in (
        ("supervisor", load_supervisor_settings, get_llm_api_key),
        ("coding", load_coding_settings, get_coding_api_key),
        ("planner", load_planner_settings, get_llm_api_key),
    ):
        from medbenchagent.runtime.model_services import resolve_model_service

        settings = loader(config_path)
        model = resolve_model_service(agent.capitalize(), settings.llm.base_url,
                                      settings.llm.model, credential())
        snapshots[agent] = replace(settings, llm=replace(settings.llm, model=model)).snapshot()
        names = ("discover_scenarios", "design_leaves") if agent == "planner" else (agent,)
        instructions.update({name: prompts.freeze_composition(name) for name in names})
    professional = {}
    for role, mode in (("data_processing", data_processing_mode), ("generation", generation_mode), ("review", review_mode)):
        if mode == "llm":
            from medbenchagent.runtime.model_services import resolve_model_service
            settings = load_work_agent_settings(role, config_path)
            model = resolve_model_service(role, settings.llm.base_url, settings.llm.model, get_llm_api_key())
            professional[role] = replace(settings, llm=replace(settings.llm, model=model)).snapshot()
            names = {"data_processing": ("profile", "prepare_task_data", "coding_handoff"),
                     "generation": ("generate",), "review": ("review_plan", "review_candidate")}[role]
            instructions.update({name: prompts.freeze_composition(name) for name in names})
    instructions["structured_repair"] = prompts.freeze_composition("structured_repair")
    if source_manifest is None and dataset_manifest is None:
        raise ValueError("Supply a source_manifest or dataset_manifest; demo data is explicit")
    if repair_limits is None:
        import tomllib
        from medbenchagent.settings import DEFAULT_CONFIG_PATH
        values = tomllib.loads((config_path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
        repair_limits = RepairLimits.model_validate(values.get("compilation", {}).get("repair_limits", {}))
    options = RunOptions(
        compilation_id=compilation_id, goal=goal,
        goal_spec=goal_spec or GoalSpec(), repair_limits=repair_limits,
        implementation_digest=implementation_digest(),
        planner_settings=snapshots["planner"],
        dataset_manifest=str(dataset_manifest.resolve()) if dataset_manifest is not None else None,
        source_manifest=str(source_manifest.resolve()) if source_manifest is not None else None,
        data_processing_mode=data_processing_mode, generation_mode=generation_mode, review_mode=review_mode,
        delivery_enabled=delivery_enabled, professional_settings=professional,
        benchmark_request=load_compilation_target(config_path, sample_count=sample_count) if generation_mode == "llm" else None,
        supervisor_settings=snapshots["supervisor"], coding_settings=snapshots["coding"],
        prompts=instructions,
    )
    directory.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(options.model_dump_json(indent=2) + "\n")
    return options


def resume_run(directory: Path, compilation_id: str) -> RunOptions:
    options = RunOptions.model_validate_json((directory / "run.json").read_text(encoding="utf-8"))
    if options.implementation_digest != implementation_digest():
        raise ValueError("This run uses incompatible code/contracts/tools. Start a new run; existing files are left untouched.")
    if options.compilation_id != compilation_id:
        raise ValueError("The directory belongs to another Compilation")
    if not (directory / "checkpoints.sqlite").is_file():
        raise ValueError("No saved checkpoint exists")
    if options.prompts:
        # Credentials may rotate; neither live settings nor prompt resources are read.
        load_project_env()
    return options


def implementation_digest():
    """Conservative code/tool compatibility check; prompt Markdown is frozen separately."""
    import hashlib
    import importlib.metadata
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    for package in ("langgraph", "langchain", "langchain-core", "langchain-openai", "pydantic"):
        digest.update(f"{package}:{importlib.metadata.version(package)}".encode())
    return digest.hexdigest()
