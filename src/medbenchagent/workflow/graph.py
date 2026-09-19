"""Supervisor orchestration with explicit dataset-local child state and global allocation."""

import json
from pathlib import Path
from typing import Any, TypedDict
from langchain_core.messages import convert_to_messages
from langgraph.graph import StateGraph, START, END
from medbenchagent.agents.generation import compile_generation_agent, run_generation
from medbenchagent.agents.audit import run_audit
from medbenchagent.agents.task_planning import compile_task_planning_agent
from medbenchagent.agents.allocation import compile_allocation_agent
from medbenchagent.planning.dataset_selection import select_datasets as run_selection
from medbenchagent.planning.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.tool_contracts import FeedbackArguments, PrepareDataArguments, DatasetId
from medbenchagent.runtime.run_artifacts import initialize_run, atomic_write_json
from medbenchagent.runtime.run_export import export_benchmark
from medbenchagent.settings import get_settings, run_settings
from medbenchagent.runtime.dependencies import publish, require, initialize_dependencies, verify_resume


class DatasetArguments(FeedbackArguments):
    dataset_id: DatasetId


class BenchmarkInputState(TypedDict, total=False):
    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    question_temperature: float
    prepare_dataset_id: str


class BenchmarkState(BenchmarkInputState, total=False):
    context: dict[str, Any]
    result: dict[str, Any]
    run_id: str
    run_directory: str
    working_blueprint_path: str
    construction_events_path: str
    benchmark_info: dict
    selected_dataset_ids: list[str]
    dataset_task_plans: dict
    raw_output_path: str
    audit_artifacts: dict
    manifest_path: str
    benchmark_id: str
    generation_shortfalls: dict
    planning_shortfall: int
    total_shortfall: int
    item_count: int
    task_counts: dict
    status: str
    summary: str
    prepared_entries: dict


class BenchmarkOutputState(TypedDict, total=False):
    run_id: str
    run_directory: str
    working_blueprint_path: str
    construction_events_path: str
    benchmark_info: dict
    selected_dataset_ids: list[str]
    dataset_task_plans: dict
    raw_output_path: str
    audit_artifacts: dict
    manifest_path: str
    benchmark_id: str
    generation_shortfalls: dict
    planning_shortfall: int
    total_shortfall: int
    item_count: int
    task_counts: dict
    status: str
    summary: str
    prepared_entries: dict


def specify(context):
    """Extract the benchmark brief from the user's complete requirement."""
    info = extract_benchmark_info(context["user_requirement"], context.get("benchmark_info"))
    if missing := missing_fields(info):
        return result("success", "Brief needs clarification: " + ", ".join(missing), benchmark_info=info, brief_ready=False)
    path = Path(context["operation_directory"]) / "blueprint.json"
    write_working_blueprint(info, path)
    return result("success", "Benchmark brief extracted", benchmark_info=info, brief_ready=True,
                  working_blueprint_path=str(path),
                  **publish(context, "Requirement", {"path": str(path), "brief": info}, ()))


COMMON = ("run_id", "run_directory", "construction_events_path", "effective_config_path", "artifact_versions",
          "limits", "user_requirement", "benchmark_info", "seed", "max_tasks_per_image", "question_temperature",
          "registration_catalog_path")


def child_context(context, dataset_id, role):
    if dataset_id not in context.get("selected_dataset_ids", []):
        raise ValueError("Dataset is not selected")
    require(context, "DatasetSelection")
    saved = context.get("child_states", {}).get(dataset_id, {}).get(role)
    child = json.loads(Path(saved).read_text(encoding="utf-8")) if saved else {}
    child.update({k: context[k] for k in COMMON if k in context})
    asset = context["dataset_assets"][dataset_id]
    child.update(dataset_scope=dataset_id, selected_dataset_id=dataset_id,
                 annotation_path=asset["access"]["annotation_path"], dataset_entry=asset["entry"],
                 dataset_access_manifest=asset["access"], dataset_card_summary=asset["card_summary"],
                 operation_directory=str(Path(context["operation_directory"]) / role / dataset_id))
    return child


def child_result(context, dataset_id, role, output):
    child = output["context"]
    path = Path(context["operation_directory"]) / f"{role}-{dataset_id}-state.json"
    atomic_write_json(path, child)
    states = {k: dict(v) for k, v in context.get("child_states", {}).items()}
    states.setdefault(dataset_id, {})[role] = str(path)
    updates = {"child_states": states, "artifact_versions": child.get("artifact_versions", context["artifact_versions"]),
               "last_agent_result": output["result"]}
    if child.get("global_issue"):
        updates["global_issue"] = child["global_issue"]
    outcome = dict(output["result"])
    outcome["updates"] = updates
    if outcome["status"] in {"complete", "needs_input"}:
        outcome["status"] = "success"
    return outcome


def describe(context):
    return {key: context.get(key) for key in (
        "user_requirement", "benchmark_info", "brief_ready", "prepare_dataset_id", "selected_dataset_ids",
        "dataset_task_plans", "dataset_generations", "dataset_audits", "allocation_path", "prepared_entries",
        "artifact_versions", "global_issue", "last_agent_result")}


def compile_benchmark_graph(*, checkpointer=None, preparation_only=False):
    planning = compile_task_planning_agent()
    generation = compile_generation_agent()
    allocation = compile_allocation_agent()

    def select_datasets(context):
        """Choose prepared datasets; escalate missing candidates for user clarification."""
        outcome = run_selection(context)
        if outcome["status"] == "needs_input":
            outcome["status"] = "success"
        return outcome

    def plan(context, *, dataset_id, feedback=""):
        """Plan tasks for exactly one selected prepared dataset; no quotas are assigned here."""
        child = child_context(context, dataset_id, "planning")
        child["feedback"] = feedback
        output = planning.invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "planning", output)
        if output["result"]["status"] == "complete":
            outcome["updates"]["dataset_task_plans"] = {**context.get("dataset_task_plans", {}), dataset_id: output["context"]["working_blueprint_path"]}
        return outcome

    def prepare_data(context, *, sources, dataset_id, feedback=""):
        """Prepare/register ONE dataset; source files must belong to that same logical dataset."""
        from medbenchagent.agents.data_pipeline import initialize
        from medbenchagent.agents.data_processing import compile_data_processing_agent
        if preparation_only and dataset_id != context["prepare_dataset_id"]:
            raise ValueError("Use the dataset_id supplied to prepare")
        child = {k: context[k] for k in COMMON if k in context}
        child["operation_directory"] = context["operation_directory"]
        child = initialize(child, sources, dataset_id)
        child["feedback"] = feedback
        output = compile_data_processing_agent().invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "preparation", output)
        if output["result"]["status"] == "complete":
            outcome["updates"]["prepared_entries"] = {**context.get("prepared_entries", {}), dataset_id: output["context"]["dataset_entry"]}
        return outcome

    def allocate(context, *, feedback=""):
        """Delegate global Concept-first allocation after every selected dataset plan is complete."""
        child = {k: context[k] for k in (*COMMON, "selected_dataset_ids", "dataset_task_plans", "operation_directory") if k in context}
        child["feedback"] = feedback
        from medbenchagent.agents.allocation import _plans
        if not any(leaf["eligible_count"] for p in _plans(child).values() for leaf in p["tasks"].values()):
            return result("blocked", "All selected datasets completed planning without eligible tasks")
        output = allocation.invoke({"context": child})
        updates = {k: output["context"][k] for k in ("artifact_versions", "allocation_path", "global_issue") if k in output["context"]}
        updates["last_agent_result"] = output["result"]
        return {**output["result"], "status": "success" if output["result"]["status"] in {"complete", "needs_input"} else output["result"]["status"], "updates": updates}

    def generate(context, *, dataset_id, feedback=""):
        """Generate one dataset's positive allocation using its own Protocols and annotation."""
        require(context, "Allocation")
        child = child_context(context, dataset_id, "generation")
        require(child, "Plan")
        assigned = json.loads(Path(context["allocation_path"]).read_text(encoding="utf-8"))
        counts = assigned["counts"][dataset_id]
        if not sum(counts.values()):
            return result("success", "Skipped dataset with zero allocation")
        blueprint = json.loads(Path(context["dataset_task_plans"][dataset_id]).read_text(encoding="utf-8"))
        for key, leaf in blueprint["tasks"].items():
            leaf["sample_count"] = counts[key]
        blueprint["allocation"] = {"allocated_count": sum(counts.values()), "counts": counts}
        path = Path(child["operation_directory"]) / "request_blueprint.json"
        atomic_write_json(path, blueprint)
        child["working_blueprint_path"] = str(path)
        outcome = run_generation(child, generation, feedback=feedback)
        child.update(outcome["updates"])
        generations = {**context.get("dataset_generations", {}), dataset_id: {
            "raw_output_path": child["raw_output_path"], "working_blueprint_path": child["working_blueprint_path"]}}
        return result("success", "Dataset generated", dataset_generations=generations, artifact_versions=child["artifact_versions"])

    def audit(context, *, dataset_id, feedback=""):
        """Audit one dataset against its original annotations and assigned tasks."""
        child = child_context(context, dataset_id, "audit")
        child.update(context["dataset_generations"][dataset_id])
        outcome = run_audit(child)
        audits = {**context.get("dataset_audits", {}), dataset_id: outcome["updates"]["audit_artifacts"]}
        return result("success", "Dataset audited", dataset_audits=audits,
                      artifact_versions=outcome["updates"]["artifact_versions"], audit_summary=outcome["updates"]["audit_summary"])

    def finish(context):
        if preparation_only:
            if context["prepare_dataset_id"] not in context.get("prepared_entries", {}):
                raise ValueError("Prepare and register the dataset first")
            return result("complete", "Dataset prepared and registered")
        return result("complete", "Benchmark completed; deficits are reported", **export_benchmark(context))

    def available_stages(context):
        if preparation_only:
            return {"prepare_data", "ask_user", "finish"}
        tools = {"specify", "ask_user", "prepare_data"}
        if not context.get("brief_ready"):
            return tools
        tools.add("select_datasets")
        try:
            require(context, "DatasetSelection")
        except ValueError:
            return tools
        tools.add("plan")
        try:
            require(context, *(f"dataset/{d}/Plan" for d in context["selected_dataset_ids"]))
        except ValueError:
            return tools
        tools.add("allocate")
        try:
            require(context, "Allocation")
        except ValueError:
            return tools
        tools.add("generate")
        assigned = json.loads(Path(context["allocation_path"]).read_text(encoding="utf-8"))
        active = [d for d, counts in assigned["counts"].items() if sum(counts.values())]
        for dataset in active:
            try:
                require(context, f"dataset/{dataset}/Generation")
                tools.add("audit")
            except ValueError:
                pass
        try:
            require(context, *(f"dataset/{d}/Audit" for d in active))
        except ValueError:
            return tools
        return tools | {"finish"}

    instructions = (
        "Prepare and register the one requested medical dataset. Interpret the user's source descriptions into "
        "prepare_data arguments; ask_user about ambiguous paths or source semantics. Never combine independent datasets. "
        "Finish only after registration succeeds. " if preparation_only else
        "Build the requested Medical VLM benchmark. Extract the brief, then select_datasets using prepared Cards. "
        "If the user supplies new raw data, prepare_data separately for each logical dataset first. Never combine "
        "independent datasets into one annotation. Plan each selected dataset explicitly by dataset_id. Once ALL plans "
        "are complete, allocate globally; then generate and audit each dataset with positive allocation. "
        "Zero allocation or a completed unsupported plan is valid; a failed/blocked plan is not. "
        "Finish to merge audited samples. Never silently lower the target or omit requested datasets. "
        "Clarify global_issue through ask_user, then re-extract the complete brief. No routine confirmations. "
        "Finite data exhaustion may finish with explicit deficits; no automatic refill loop. "
    )
    supervisor = compile_agent("SupervisorAgent", instructions,
        {"prepare_data": prepare_data} if preparation_only else
        {"specify": specify, "select_datasets": select_datasets, "prepare_data": prepare_data,
         "plan": plan, "allocate": allocate, "generate": generate, "audit": audit},
        describe, finish, available_tools=available_stages,
        args_schemas={"prepare_data": PrepareDataArguments} if preparation_only else
        {"prepare_data": PrepareDataArguments, "plan": DatasetArguments, "allocate": FeedbackArguments,
         "generate": DatasetArguments, "audit": DatasetArguments})

    def initialize(state):
        overrides = {k: state[k] for k in ("seed", "max_tasks_per_image", "question_temperature") if k in state}
        requirement = state.get("user_requirement") or next((getattr(m, "content", "") for m in reversed(convert_to_messages(state.get("messages", []))) if getattr(m, "type", "") == "human"), "")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("A natural-language requirement is required")
        settings = get_settings()
        defaults = {k: getattr(settings.generation, k) for k in ("seed", "max_tasks_per_image", "question_temperature")}
        run = initialize_run(graph_overrides=overrides)
        context = dict(run, **(defaults | overrides), user_requirement=requirement,
                       operation_directory=str(Path(run["run_directory"]) / ".runtime"), limits=settings.harness,
                       registration_catalog_path=str(settings.paths.dataset_catalog),
                       prepare_dataset_id=state.get("prepare_dataset_id"))
        initialize_dependencies(context, settings)
        return dict(run, context=context)

    def run_supervisor(state):
        verify_resume(state["context"])
        with run_settings(state["context"]["effective_config_path"]):
            output = supervisor.invoke({"context": state["context"]})
        public = {k: v for k, v in output["context"].items() if k in BenchmarkState.__annotations__ and k not in {"context", "result"}}
        return dict(public, context=output["context"], result=output["result"],
                    status=output["result"]["status"], summary=output["result"]["summary"])

    graph = StateGraph(BenchmarkState, input_schema=BenchmarkInputState, output_schema=BenchmarkOutputState)
    graph.add_node("initialize", initialize)
    graph.add_node("supervisor", run_supervisor)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile(checkpointer=checkpointer)


benchmark_graph = compile_benchmark_graph()
