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
from medbenchagent.runtime.tool_contracts import FeedbackArguments, PrepareDataArguments, DatasetId, DelegationArguments, RunIntent, PageArguments, NoArguments
from medbenchagent.data.readiness import list_datasets, inspect_dataset_status, dataset_status, configured_source, reuse_prepared
from medbenchagent.runtime.run_artifacts import initialize_run, atomic_write_json
from medbenchagent.runtime.run_export import export_benchmark
from medbenchagent.settings import get_settings, run_settings
from medbenchagent.runtime.dependencies import publish, require, initialize_dependencies, verify_resume
from medbenchagent.runtime.preflight import preflight_errors, rag_capability


class DatasetArguments(FeedbackArguments):
    dataset_id: DatasetId


class DatasetStatusArguments(NoArguments):
    dataset_id: DatasetId


class PlanningArguments(DelegationArguments):
    dataset_id: DatasetId


class BenchmarkInputState(TypedDict, total=False):
    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    question_temperature: float
    prepare_dataset_id: str
    entry_intent: str
    requested_dataset_ids: list[str]


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
    if context.get("brief_requirement") == context["user_requirement"]:
        raise ValueError("This requirement has already been extracted; clarify missing fields first")
    info = extract_benchmark_info(context["user_requirement"])
    source = "user"
    if info.get("sample_count") is None:
        info["sample_count"] = get_settings().benchmark.default_sample_count
        source = "configuration"
    missing = missing_fields(info)
    updates = {"benchmark_info": info, "brief_ready": not missing, "brief_missing_fields": missing,
               "brief_requirement": context["user_requirement"], "sample_count_source": source}
    if missing:
        return result("success", "Brief needs clarification: " + ", ".join(missing), **updates)
    path = Path(context["operation_directory"]) / "blueprint.json"
    write_working_blueprint(info, path)
    return result("success", "Benchmark brief extracted", **updates,
                  working_blueprint_path=str(path),
                  **publish(context, "Requirement", {"path": str(path), "brief": info,
                             "user_requirement": context["user_requirement"], "sample_count_source": source}, ()))


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
    details = {key: context.get(key) for key in (
        "user_requirement", "benchmark_info", "brief_ready", "brief_missing_fields", "sample_count_source", "prepare_dataset_id", "selected_dataset_ids",
        "dataset_task_plans", "dataset_generations", "dataset_audits", "allocation_path", "prepared_entries",
        "artifact_versions", "global_issue", "last_agent_result", "entry_intent", "run_intent", "requested_dataset_ids")}
    ids = set(context.get("selected_dataset_ids", [])) | set(context.get("run_intent", {}).get("prepare_dataset_ids", []))
    details["dataset_readiness"] = {d: {kind: _current(context, f"dataset/{d}/{kind}")
                                      for kind in ("Dataset", "Plan", "Generation", "Audit")} for d in sorted(ids)}
    details["outstanding_work"] = {"prepare_dataset_ids": preparation_pending(context),
                                   **pending_datasets(context),
                                   "available_actions": sorted(available_stages(context))}
    details["rag_capability"] = context.get("rag_capability")
    return details


def set_intent(context, *, intent, prepare_dataset_ids):
    """Record user intent and explicitly requested preparation targets; clarify unknown IDs first."""
    value = RunIntent(intent=intent, prepare_dataset_ids=prepare_dataset_ids).model_dump()
    if context.get("entry_intent") == "prepare_dataset" and intent != "prepare_dataset":
        raise ValueError("The prepare command only prepares datasets")
    if context.get("entry_intent") == "build_benchmark" and intent == "prepare_dataset":
        raise ValueError("The build command must build a benchmark")
    if context.get("prepare_dataset_id") and prepare_dataset_ids != [context["prepare_dataset_id"]]:
        raise ValueError("Use the dataset ID supplied to prepare")
    if intent != "prepare_dataset":
        if errors := preflight_errors(get_settings(), build=True):
            return result("blocked", "Build preflight failed: " + "; ".join(errors))
    updates = {"run_intent": value}
    if intent == "prepare_then_build" and context.get("requested_dataset_ids") is not None:
        requested = list(dict.fromkeys(context["requested_dataset_ids"] + prepare_dataset_ids))
        updates["requested_dataset_ids"] = requested
        if requested != context["requested_dataset_ids"]:
            updates.update(publish(context, "PreparedCandidates",
                                   {"entries": context.get("prepared_entries", {}), "requested_dataset_ids": requested}, ()))
    return result("success", "Run intent recorded", **updates)


def _current(context, *kinds):
    try:
        require(context, *kinds)
        return True
    except ValueError:
        return False


def preparation_pending(context):
    return [d for d in context.get("run_intent", {}).get("prepare_dataset_ids", [])
            if d not in context.get("prepared_entries", {}) or not _current(context, f"dataset/{d}/Dataset")]


def pending_datasets(context):
    pending = {"pending_plan_dataset_ids": [], "pending_generation_dataset_ids": [], "pending_audit_dataset_ids": []}
    if not _current(context, "DatasetSelection"):
        return pending
    pending["pending_plan_dataset_ids"] = [d for d in context["selected_dataset_ids"] if not _current(context, f"dataset/{d}/Plan")]
    if not _current(context, "Allocation"):
        return pending
    assigned = json.loads(Path(context["allocation_path"]).read_text(encoding="utf-8"))
    for dataset_id in context["selected_dataset_ids"]:
        if not sum(assigned["counts"][dataset_id].values()):
            continue
        if not _current(context, f"dataset/{dataset_id}/Generation"):
            pending["pending_generation_dataset_ids"].append(dataset_id)
        elif not _current(context, f"dataset/{dataset_id}/Audit"):
            pending["pending_audit_dataset_ids"].append(dataset_id)
    return pending


def require_pending(context, stage, dataset_id):
    if dataset_id not in pending_datasets(context)[f"pending_{stage}_dataset_ids"]:
        raise ValueError(f"Dataset {dataset_id} is not pending {stage}")


def available_stages(context):
    tools = {"set_intent", "ask_user", "list_datasets", "inspect_dataset_status"}
    intent = context.get("run_intent")
    if not intent:
        return tools
    pending = preparation_pending(context)
    if intent["intent"] == "prepare_dataset":
        return tools | ({"finish"} if not pending else {"prepare_data"})
    if context.get("brief_requirement") != context.get("user_requirement") or "brief_requirement" not in context:
        tools.add("specify")
    if not context.get("brief_ready"):
        return tools
    if pending:
        return tools | {"prepare_data"}
    if not _current(context, "DatasetSelection"):
        return tools | {"select_datasets"}
    pending = pending_datasets(context)
    if pending["pending_plan_dataset_ids"]:
        return tools | {"plan"}
    if not _current(context, "Allocation"):
        return tools | {"allocate"}
    if pending["pending_generation_dataset_ids"]:
        tools.add("generate")
    if pending["pending_audit_dataset_ids"]:
        tools.add("audit")
    if not pending["pending_generation_dataset_ids"] and not pending["pending_audit_dataset_ids"]:
        tools.add("finish")
    return tools


def compile_benchmark_graph(*, checkpointer=None):
    planning = compile_task_planning_agent()
    generation = compile_generation_agent()
    allocation = compile_allocation_agent()

    def select_datasets(context):
        """Select prepared data or route matching configured sources into preparation without asking for paths."""
        outcome = run_selection(context)
        targets = outcome["updates"].get("preparation_required", [])
        if targets:
            context.update(outcome["updates"])
            intent_result = set_intent(context, intent="prepare_then_build", prepare_dataset_ids=targets)
            if intent_result["status"] == "blocked":
                return {**intent_result, "updates": outcome["updates"]}
            outcome["updates"].update(intent_result["updates"])
        return outcome

    def plan(context, *, dataset_id, delegation):
        """Plan tasks for exactly one selected prepared dataset; no quotas are assigned here."""
        require_pending(context, "plan", dataset_id)
        child = child_context(context, dataset_id, "planning")
        child["delegation"] = delegation
        publish(child, "PlanningDelegation", delegation, ())
        output = planning.invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "planning", output)
        if output["result"]["status"] == "complete":
            outcome["updates"]["dataset_task_plans"] = {**context.get("dataset_task_plans", {}), dataset_id: output["context"]["working_blueprint_path"]}
        return outcome

    def prepare_data(context, *, dataset_id, mode="reuse_or_prepare", feedback=""):
        """Reuse valid prepared data or prepare configured sources. Use rebuild only for an explicit user request."""
        from medbenchagent.agents.data_pipeline import initialize
        from medbenchagent.agents.data_processing import compile_data_processing_agent
        if dataset_id not in preparation_pending(context):
            raise ValueError("Dataset is not an outstanding preparation target; update run intent first")
        operation = Path(context["operation_directory"])
        if mode == "reuse_or_prepare" and (operation / "reused_dataset.json").is_file():
            return reuse_prepared(context, dataset_id)
        status = dataset_status(context, dataset_id)
        saved = context.get("child_states", {}).get(dataset_id, {}).get("preparation")
        if not saved and not (operation / "preparation.json").is_file() and mode == "reuse_or_prepare":
            if status["prepared"]:
                return reuse_prepared(context, dataset_id)
            if status["available"] is not None:
                return result("blocked", "Existing dataset needs repair or explicit rebuild: " + "; ".join(status["issues"]))
        if status["source_errors"]:
            return result("blocked", "; ".join(status["source_errors"]))
        sources = configured_source(dataset_id)["sources"]
        if saved:
            from medbenchagent.data.contracts import Source
            child = json.loads(Path(saved).read_text(encoding="utf-8"))
            if child["sources"] != [Source.model_validate(s).model_dump() for s in sources]:
                raise ValueError("Preparation sources changed; start a new experiment")
            if child["preparation_mode"] != mode:
                raise ValueError("Preparation mode changed; resume with the original mode")
            child.update({k: context[k] for k in COMMON if k in context})
            child["operation_directory"] = context["operation_directory"]
        else:
            child = {k: context[k] for k in COMMON if k in context}
            child["operation_directory"] = context["operation_directory"]
            child = initialize(child, sources, dataset_id)
            child["preparation_mode"] = mode
        child["feedback"] = feedback
        output = compile_data_processing_agent().invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "preparation", output)
        if output["result"]["status"] == "complete":
            entries = {**context.get("prepared_entries", {}), dataset_id: output["context"]["dataset_entry"]}
            outcome["updates"]["prepared_entries"] = entries
            published = dict(context, artifact_versions=outcome["updates"]["artifact_versions"])
            outcome["updates"].update(publish(published, "PreparedCandidates", {"entries": entries}, ()))
        return outcome

    def allocate(context, *, delegation):
        """Delegate global Concept-first allocation after every selected dataset plan is complete."""
        child = {k: context[k] for k in (*COMMON, "selected_dataset_ids", "dataset_task_plans", "operation_directory") if k in context}
        child["delegation"] = delegation
        publish(child, "AllocationDelegation", delegation, ())
        from medbenchagent.agents.allocation import _plans
        if not any(leaf["eligible_count"] for p in _plans(child).values() for leaf in p["tasks"].values()):
            return result("blocked", "All selected datasets completed planning without eligible tasks")
        output = allocation.invoke({"context": child})
        updates = {k: output["context"][k] for k in ("artifact_versions", "allocation_path", "global_issue") if k in output["context"]}
        updates["last_agent_result"] = output["result"]
        return {**output["result"], "status": "success" if output["result"]["status"] in {"complete", "needs_input"} else output["result"]["status"], "updates": updates}

    def generate(context, *, dataset_id, feedback=""):
        """Generate one dataset's positive allocation using its own Protocols and annotation."""
        require_pending(context, "generation", dataset_id)
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
        require_pending(context, "audit", dataset_id)
        child = child_context(context, dataset_id, "audit")
        child.update(context["dataset_generations"][dataset_id])
        outcome = run_audit(child)
        audits = {**context.get("dataset_audits", {}), dataset_id: outcome["updates"]["audit_artifacts"]}
        return result("success", "Dataset audited", dataset_audits=audits,
                      artifact_versions=outcome["updates"]["artifact_versions"], audit_summary=outcome["updates"]["audit_summary"])

    def finish(context):
        if context["run_intent"]["intent"] == "prepare_dataset":
            if preparation_pending(context):
                raise ValueError("Prepare and register the dataset first")
            return result("complete", "Dataset prepared and registered")
        return result("complete", "Benchmark completed; deficits are reported", **export_benchmark(context))

    supervisor = compile_agent("SupervisorAgent",
        "Coordinate the user's medical dataset preparation or benchmark construction. "
        "Set run intent from the request when absent; update it when the user adds preparation targets. "
        "Observe list_datasets and inspect_dataset_status before preparation. Source paths come only from config; "
        "never ask the user for paths or invent dataset IDs. Default to reuse_or_prepare; rebuild only when explicitly requested. "
        "Respect the entry intent. Prepare each outstanding logical dataset independently, never merge datasets. "
        "Preparation-only work finishes after all requested registrations; no benchmark brief is needed. "
        "For construction, extract benchmark-level scope, prepare requested new data, then select prepared datasets "
        "with high recall. Plan each selected dataset, allocate after every plan is current, generate and audit "
        "positive allocations, then export. Follow artifact readiness, not a fixed script. "
        "Delegate explicit objectives, constraints, preferences and revision notes to Planning and Allocation. "
        "Delegations must respect the original user requirement; never invent a task focus. "
        "Unsupported plans and zero allocation are valid, failures are not. Never omit explicitly requested datasets. "
        "Clarify global_issue through ask_user, then re-extract the complete brief. No routine confirmations. "
        "Finite exhaustion may finish with reported deficits; no automatic refill loop.",
        {"set_intent": set_intent, "specify": specify, "select_datasets": select_datasets,
         "list_datasets": list_datasets, "inspect_dataset_status": inspect_dataset_status,
         "prepare_data": prepare_data, "plan": plan, "allocate": allocate, "generate": generate, "audit": audit},
        describe, finish, available_tools=available_stages,
        args_schemas={"set_intent": RunIntent, "prepare_data": PrepareDataArguments,
                      "list_datasets": PageArguments, "inspect_dataset_status": DatasetStatusArguments,
                      "plan": PlanningArguments, "allocate": DelegationArguments,
                      "generate": DatasetArguments, "audit": DatasetArguments})

    def initialize(state):
        overrides = {k: state[k] for k in ("seed", "max_tasks_per_image", "question_temperature") if k in state}
        requirement = state.get("user_requirement") or next((getattr(m, "content", "") for m in reversed(convert_to_messages(state.get("messages", []))) if getattr(m, "type", "") == "human"), "")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("A natural-language requirement is required")
        settings = get_settings()
        if errors := preflight_errors(settings, build=state.get("entry_intent") == "build_benchmark"):
            raise ValueError("Local preflight failed: " + "; ".join(errors))
        defaults = {k: getattr(settings.generation, k) for k in ("seed", "max_tasks_per_image", "question_temperature")}
        run = initialize_run(graph_overrides=overrides)
        context = dict(run, **(defaults | overrides), user_requirement=requirement,
                       operation_directory=str(Path(run["run_directory"]) / ".runtime"), limits=settings.harness,
                       registration_catalog_path=str(settings.paths.dataset_catalog),
                       rag_capability=rag_capability(settings),
                       prepare_dataset_id=state.get("prepare_dataset_id"),
                       entry_intent=state.get("entry_intent"),
                       requested_dataset_ids=state.get("requested_dataset_ids"))
        if state.get("prepare_dataset_id"):
            context["entry_intent"] = "prepare_dataset"
            context["run_intent"] = RunIntent(intent="prepare_dataset", prepare_dataset_ids=[state["prepare_dataset_id"]]).model_dump()
        elif context["entry_intent"] == "build_benchmark":
            context["run_intent"] = RunIntent(intent="build_benchmark").model_dump()
        initialize_dependencies(context, settings)
        publish(context, "PreparedCandidates", {"entries": {}}, ())
        return dict(run, context=context)

    def run_supervisor(state):
        verify_resume(state["context"])
        with run_settings(state["context"]["effective_config_path"]):
            if errors := preflight_errors(get_settings(), build=state["context"].get("run_intent", {}).get("intent") in {"build_benchmark", "prepare_then_build"}):
                raise ValueError("Local preflight failed: " + "; ".join(errors))
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
