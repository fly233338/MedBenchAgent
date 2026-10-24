"""Run-owned configuration and prompt snapshots, with no Agent assembly or legacy BIR."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any

from pydantic import model_validator, Field, TypeAdapter

from medbenchagent.agents._shared import prompting as prompts
from medbenchagent.settings import (
    CodingSettings, SupervisorSettings, TaskPlannerSettings, get_coding_api_key, get_llm_api_key,
    load_coding_settings, load_project_env, load_supervisor_settings, load_planner_settings,
    WorkAgentSettings, load_work_agent_settings, load_compilation_target,
)

from .contracts import Contract, Identifier, Text, GoalSpec, RepairLimits
from medbenchagent.agents.data_processing.coding.skills import FrozenSkills, freeze_skills, verify_skills


def required_roles(goal: GoalSpec, *, raw_source: bool) -> set[str]:
    if goal.kind == "profile" and not raw_source:
        raise ValueError("Profile goals require a SourceManifest")
    roles = {"supervisor"}
    if goal.kind != "profile":
        roles.add("planner")
    if raw_source or goal.kind == "benchmark" or goal.plan_level == "executable_leaves":
        roles.update({"data_processing", "coding"})
    if goal.kind == "benchmark":
        roles.update({"generation", "review", "delivery"})
    return roles


class RunOptions(Contract):
    compilation_id: Identifier
    runtime_directory: str
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    goal: Text
    goal_spec: GoalSpec = Field(default_factory=GoalSpec)
    repair_limits: RepairLimits = Field(default_factory=RepairLimits)
    implementation_digest: str
    dataset_manifest: str | None = None
    source_manifest: str | None = None
    professional_settings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    benchmark_request: dict[str, Any] | None = None
    planner_settings: dict[str, Any] | None = None
    supervisor_settings: dict[str, Any]
    coding_settings: dict[str, Any] | None = None
    coding_skills: FrozenSkills | None = None
    prompts: dict[str, prompts.PromptSnapshot]

    @model_validator(mode="after")
    def check_selected_implementations(self):
        if not Path(self.runtime_directory).is_absolute():
            raise ValueError("runtime_directory must be absolute")
        required = set()
        if (self.source_manifest is None) == (self.dataset_manifest is None):
            raise ValueError("Select exactly one source or canonical Dataset")
        roles = required_roles(self.goal_spec, raw_source=self.source_manifest is not None)
        enabled = roles & {"data_processing", "generation", "review"}
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
            if agent not in roles:
                if snapshot is not None:
                    raise ValueError(f"Unused {agent} configuration")
                continue
            if snapshot is None:
                raise ValueError(f"Missing frozen {agent} configuration")
            if not schema.from_snapshot(snapshot).llm.model:
                raise ValueError(f"Missing frozen {agent} model identity")
            required.update({"planner": {"discover_scenarios", "design_leaves"}}.get(agent, {agent}))
        required.add("structured_repair")
        if set(self.prompts) != required:
            raise ValueError("Prompt snapshots must match the selected model-driven agents")
        if (self.benchmark_request is not None) != (self.goal_spec.kind == "benchmark"):
            raise ValueError("Only benchmark goals freeze a quantity request")
        if "coding" in roles:
            if self.coding_skills is None or self.coding_skills.names != self.coding_settings["coding"]["skill_names"]:
                raise ValueError("Frozen skills must match Coding configuration")
        elif self.coding_skills is not None:
            raise ValueError("Unused Coding skills")
        return self


def create_run(
    directory: Path, *, compilation_id: str, goal: str,
    config_path: Path | None = None, goal_spec: GoalSpec | None = None, repair_limits: RepairLimits | None = None,
    dataset_manifest: Path | None = None,
    source_manifest: Path | None = None,
    sample_count: int | None = None,
    runtime_directory: Path | None = None,
    language: str | None = None,
) -> RunOptions:
    compilation_id = TypeAdapter(Identifier).validate_python(compilation_id)
    directory = directory.resolve()
    path = directory / "run.json"
    from medbenchagent.settings import load_paths
    from medbenchagent.runtime.run_artifacts import separate_directories
    runtime_directory = (runtime_directory or load_paths(config_path)["runtime_root"] / compilation_id).resolve()
    separate_directories(directory, runtime_directory)
    goal_spec = goal_spec or GoalSpec()
    if (source_manifest is None) == (dataset_manifest is None):
        raise ValueError("Supply exactly one source_manifest or dataset_manifest")
    roles = required_roles(goal_spec, raw_source=source_manifest is not None)
    if path.exists() or (runtime_directory / "checkpoints.sqlite").exists():
        raise ValueError("This directory already contains a run; use resume")
    if sample_count is not None and (type(sample_count) is not int or sample_count <= 0 or "generation" not in roles):
        raise ValueError("sample_count requires enabled Generation and a positive integer")
    if source_manifest is not None:
        from medbenchagent.data.task_data import SourceManifest
        source = SourceManifest.model_validate_json(source_manifest.read_bytes())
        for item in source.sources:
            root = (source_manifest.parent / item.root).resolve()
            if directory.is_relative_to(root):
                raise ValueError("Run output directory must not be inside a source directory")
            try:
                with os.scandir(root) as entries:
                    next(entries, None)
            except OSError as exc:
                raise ValueError(f"Source {item.source_id} directory is not readable: {root}") from exc
        separate_directories(directory, runtime_directory, [(source_manifest.parent / item.root).resolve() for item in source.sources])
        roots = {item.source_id: (source_manifest.parent / item.root).resolve() for item in source.sources}
        from medbenchagent.data.dataset import _relative_file
        for item in source.key_files:
            with _relative_file(roots[item.source_id], item.path).open("rb") as handle:
                handle.read(1)
    if dataset_manifest is not None:
        from medbenchagent.data.dataset import FileDatasetReader
        canonical = FileDatasetReader(dataset_manifest).get_manifest()
        separate_directories(directory, runtime_directory,
                             [dataset_manifest.parent, *((dataset_manifest.parent / item.root).resolve() for item in canonical.sources)])
    snapshots, instructions = {}, {}
    for agent, loader, credential in (
        ("supervisor", load_supervisor_settings, get_llm_api_key),
        ("coding", load_coding_settings, get_coding_api_key),
        ("planner", load_planner_settings, get_llm_api_key),
    ):
        if agent not in roles:
            continue
        from medbenchagent.runtime.model_services import resolve_model_service

        settings = loader(config_path)
        model = resolve_model_service(agent.capitalize(), settings.llm.base_url,
                                      settings.llm.model, credential())
        snapshots[agent] = replace(settings, llm=replace(settings.llm, model=model)).snapshot()
        names = ("discover_scenarios", "design_leaves") if agent == "planner" else (agent,)
        instructions.update({name: prompts.freeze_composition(name) for name in names})
    professional = {}
    for role in ("data_processing", "generation", "review"):
        if role in roles:
            from medbenchagent.runtime.model_services import resolve_model_service
            settings = load_work_agent_settings(role, config_path)
            model = resolve_model_service(role, settings.llm.base_url, settings.llm.model, get_llm_api_key())
            professional[role] = replace(settings, llm=replace(settings.llm, model=model)).snapshot()
            names = {"data_processing": ("profile", "prepare_task_data", "coding_handoff"),
                     "generation": ("generate",), "review": ("review_plan", "review_candidate")}[role]
            instructions.update({name: prompts.freeze_composition(name) for name in names})
    instructions["structured_repair"] = prompts.freeze_composition("structured_repair")
    if repair_limits is None:
        import tomllib
        from medbenchagent.settings import DEFAULT_CONFIG_PATH
        values = tomllib.loads((config_path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
        repair_limits = RepairLimits.model_validate(values.get("compilation", {}).get("repair_limits", {}))
    from medbenchagent.data.task_data import write_once
    if source_manifest is not None:
        from medbenchagent.data.dataset import DatasetSource
        source = source.model_copy(update={"sources": [DatasetSource(source_id=item.source_id, root=str((source_manifest.parent / item.root).resolve())) for item in source.sources]})
        source_manifest = directory / "inputs" / "sources.json"
        write_once(source_manifest, (source.model_dump_json(indent=2) + "\n").encode())
    else:
        from medbenchagent.data.dataset import snapshot_dataset
        dataset_manifest = snapshot_dataset(dataset_manifest, directory / "inputs" / "dataset")
    runtime_directory.mkdir(parents=True, exist_ok=True)
    (runtime_directory / "locks").mkdir(exist_ok=True)
    options = RunOptions(
        compilation_id=compilation_id, goal=goal, runtime_directory=str(runtime_directory),
        input_digest=hashlib.sha256((source_manifest or dataset_manifest).read_bytes()).hexdigest(),
        goal_spec=goal_spec, repair_limits=repair_limits,
        implementation_digest=implementation_digest(),
        planner_settings=snapshots.get("planner"),
        dataset_manifest=str(dataset_manifest.resolve()) if dataset_manifest is not None else None,
        source_manifest=str(source_manifest.resolve()) if source_manifest is not None else None,
        professional_settings=professional,
        benchmark_request=({**load_compilation_target(config_path, sample_count=sample_count),
                            **({"language": language.strip()} if language is not None and language.strip() else {})}
                           if "generation" in roles else None),
        supervisor_settings=snapshots["supervisor"], coding_settings=snapshots.get("coding"),
        coding_skills=freeze_skills(runtime_directory, snapshots["coding"]["coding"]["skill_names"]) if "coding" in roles else None,
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
    verify_input(options)
    if options.coding_skills is not None:
        verify_skills(Path(options.runtime_directory), options.coding_skills)
    if options.compilation_id != compilation_id:
        raise ValueError("The directory belongs to another Compilation")
    if not (Path(options.runtime_directory) / "checkpoints.sqlite").is_file():
        raise ValueError("No saved checkpoint exists")
    if options.prompts:
        # Credentials may rotate; neither live settings nor prompt resources are read.
        load_project_env()
    return options


def verify_input(options):
    from medbenchagent.data.dataset import file_digest
    if file_digest(Path(options.source_manifest or options.dataset_manifest)) != options.input_digest:
        raise ValueError("Frozen Run input manifest changed")


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
