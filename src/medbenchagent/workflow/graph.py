"""Supervisor orchestration with explicit dataset-local child state and global allocation."""

import json
from pathlib import Path
from typing import Any, TypedDict
from langchain_core.messages import convert_to_messages
from langgraph.graph import StateGraph, START, END
from medbenchagent.agents.generation import compile_generation_agent, run_generation
from medbenchagent.agents.finalization import compile_finalization_agent, restore_delivery
from medbenchagent.agents.templates import prepare_templates as run_templates
from medbenchagent.agents.task_planning import compile_task_planning_agent, apply_template_feedback
from medbenchagent.agents.allocation import compile_allocation_agent
from medbenchagent.planning.dataset_selection import select_datasets as run_selection
from medbenchagent.planning.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.tool_contracts import PrepareDataArguments, DatasetId, DelegationArguments, RunIntent, PageArguments, NoArguments
from medbenchagent.data.readiness import list_datasets, inspect_dataset_status, dataset_status, configured_source, reuse_prepared
from medbenchagent.runtime.run_artifacts import initialize_run, atomic_write_json
from medbenchagent.settings import get_settings, run_settings
from medbenchagent.runtime.dependencies import publish, require, initialize_dependencies, verify_resume
from medbenchagent.runtime.preflight import preflight_errors, rag_capability
from medbenchagent.runtime.clarifications import inject_answers, resolve_clarification, ResolveClarificationArguments


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
    raw_source_paths: dict
    delivery_path: str
    delivery_counts: dict
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
    raw_source_paths: dict
    delivery_path: str
    delivery_counts: dict
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
    sample_scope = info.pop("sample_scope", [])
    total_limit = info.pop("max_total_tasks_per_image", None)
    if info.get("modalities") and not any(r["kind"] == "modality" for r in sample_scope):
        raise ValueError("Explicit modalities must be represented in sample_scope; re-extract the requirement")
    source = "user"
    if info.get("sample_count") is None:
        info["sample_count"] = get_settings().benchmark.default_sample_count
        source = "configuration"
    missing = missing_fields(info)
    updates = {"benchmark_info": info, "sample_scope": sample_scope,
               "max_total_tasks_per_image": total_limit if total_limit is not None else context.get("configured_total_tasks_per_image"),
               "brief_ready": not missing, "brief_missing_fields": missing,
               "brief_requirement": context["user_requirement"], "sample_count_source": source}
    if missing:
        return result("success", "Brief needs clarification: " + ", ".join(missing), **updates)
    path = Path(context["operation_directory"]) / "blueprint.json"
    write_working_blueprint(info, path)
    return result("success", "Benchmark brief extracted", **updates,
                  working_blueprint_path=str(path),
                  **publish(context, "Requirement", {"path": str(path), "brief": info,
                             "user_requirement": context["user_requirement"], "sample_count_source": source,
                             "sample_scope": sample_scope, "max_total_tasks_per_image": updates["max_total_tasks_per_image"]}, ()))


COMMON = ("run_id", "run_directory", "construction_events_path", "effective_config_path", "artifact_versions",
          "limits", "user_requirement", "benchmark_info", "seed", "max_tasks_per_image", "question_temperature",
          "registration_catalog_path", "sample_scope", "max_total_tasks_per_image")


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
    return inject_answers(context, child, {"planning": "TaskPlanningAgent", "generation": "GenerationAgent",
                                          "templates": "TaskPlanningAgent"}[role], dataset_id)


def child_result(context, dataset_id, role, output):
    child = output["context"]
    path = Path(context["operation_directory"]) / f"{role}-{dataset_id}-state.json"
    atomic_write_json(path, child)
    states = {k: dict(v) for k, v in context.get("child_states", {}).items()}
    states.setdefault(dataset_id, {})[role] = str(path)
    updates = {"child_states": states, "artifact_versions": child.get("artifact_versions", context["artifact_versions"]),
               "last_agent_result": output["result"]}
    updates["issues"] = {**context.get("issues", {}), **child.get("issues", {})}
    pending = child.get("global_issue")
    if pending and updates["issues"].get(pending["issue_id"], {}).get("status") != "resolved":
        updates["global_issue"] = pending
    outcome = dict(output["result"])
    outcome["updates"] = updates
    if outcome["status"] in {"complete", "needs_input"}:
        outcome["status"] = "success"
    return outcome


def describe(context):
    details = {key: context.get(key) for key in (
        "user_requirement", "benchmark_info", "brief_ready", "brief_missing_fields", "sample_count_source", "prepare_dataset_id", "selected_dataset_ids",
        "dataset_task_plans", "dataset_templates", "template_revision_requests", "dataset_generations", "delivery_counts", "allocation_path", "prepared_entries",
        "artifact_versions", "global_issue", "last_agent_result", "entry_intent", "run_intent", "requested_dataset_ids")}
    ids = set(context.get("selected_dataset_ids", [])) | set(context.get("run_intent", {}).get("prepare_dataset_ids", []))
    details["dataset_readiness"] = {d: {kind: _current(context, f"dataset/{d}/{kind}")
                                      for kind in ("Dataset", "Plan", "Templates", "Generation")} for d in sorted(ids)}
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
    pending = {"pending_plan_dataset_ids": [], "pending_templates_dataset_ids": [], "pending_generation_dataset_ids": []}
    if not _current(context, "DatasetSelection"):
        return pending
    for dataset in context["selected_dataset_ids"]:
        feedback = context.get("template_revision_requests", {}).get(dataset)
        if not _current(context, f"dataset/{dataset}/Plan") or (feedback and feedback["plan_version"] == context["artifact_versions"].get(f"dataset/{dataset}/Plan")):
            pending["pending_plan_dataset_ids"].append(dataset)
        elif not _current(context, f"dataset/{dataset}/Templates"):
            pending["pending_templates_dataset_ids"].append(dataset)
    if not _current(context, "Allocation"):
        return pending
    assigned = json.loads(Path(context["allocation_path"]).read_text(encoding="utf-8"))
    pending["pending_generation_dataset_ids"] = [d for d in context["selected_dataset_ids"]
        if sum(assigned["counts"][d].values()) and not _current(context, f"dataset/{d}/Generation")]
    return pending


def require_pending(context, stage, dataset_id):
    if dataset_id not in pending_datasets(context)[f"pending_{stage}_dataset_ids"]:
        raise ValueError(f"Dataset {dataset_id} is not pending {stage}")


def available_stages(context):
    if context.get("global_issue"):
        return {"resolve_clarification", "ask_user"} if context["global_issue"]["status"] == "answered" else {"ask_user"}
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
    if pending["pending_plan_dataset_ids"] or pending["pending_templates_dataset_ids"]:
        if pending["pending_plan_dataset_ids"]:
            tools.add("plan")
        if pending["pending_templates_dataset_ids"]:
            tools.add("prepare_templates")
        return tools
    if not _current(context, "Allocation"):
        return tools | {"allocate"}
    if pending["pending_generation_dataset_ids"]:
        return tools | {"generate"}
    return tools | ({"finish"} if _current(context, "Delivery") else {"finalize"})


def compile_benchmark_graph(*, checkpointer=None):
    planning = compile_task_planning_agent()
    generation = compile_generation_agent()
    allocation = compile_allocation_agent()
    finalization = compile_finalization_agent()

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
        feedback = context.get("template_revision_requests", {}).get(dataset_id)
        if feedback:
            correction = apply_template_feedback(child, feedback)
            if correction["status"] == "blocked":
                return correction
            child.update(correction["updates"])
        output = planning.invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "planning", output)
        if output["result"]["status"] == "complete":
            requests = dict(context.get("template_revision_requests", {}))
            requests.pop(dataset_id, None)
            outcome["updates"]["template_revision_requests"] = requests
            outcome["updates"]["dataset_task_plans"] = {**context.get("dataset_task_plans", {}), dataset_id: output["context"]["working_blueprint_path"]}
        return outcome

    def prepare_templates(context, *, dataset_id):
        """Review templates immediately after this Dataset's Plan; return task defects to Planning."""
        require_pending(context, "templates", dataset_id)
        child = child_context(context, dataset_id, "templates")
        child["working_blueprint_path"] = context["dataset_task_plans"][dataset_id]
        outcome = run_templates(child)
        requests = dict(context.get("template_revision_requests", {}))
        if feedback := outcome["updates"].pop("template_revision_request", None):
            requests[dataset_id] = feedback
        elif outcome["status"] == "success":
            requests.pop(dataset_id, None)
            outcome["updates"]["dataset_templates"] = {**context.get("dataset_templates", {}), dataset_id: outcome["updates"].pop("templates_path")}
        outcome["updates"]["template_revision_requests"] = requests
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
        inject_answers(context, child, "DataProcessingAgent", dataset_id)
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
        saved = context.get("allocation_state_path")
        child = json.loads(Path(saved).read_text(encoding="utf-8")) if saved else {}
        child.update({k: context[k] for k in (*COMMON, "selected_dataset_ids", "dataset_task_plans", "operation_directory") if k in context})
        inject_answers(context, child, "AllocationAgent")
        child["delegation"] = delegation
        publish(child, "AllocationDelegation", delegation, ())
        from medbenchagent.agents.allocation import _plans
        if not any(leaf["eligible_count"] for p in _plans(child).values() for leaf in p["tasks"].values()):
            return result("blocked", "All selected datasets completed planning without eligible tasks")
        output = allocation.invoke({"context": child})
        state_path = Path(context["operation_directory"]) / "allocation-state.json"
        atomic_write_json(state_path, output["context"])
        updates = {k: output["context"][k] for k in ("artifact_versions", "allocation_path", "global_issue") if k in output["context"]}
        updates["last_agent_result"] = output["result"]
        updates["allocation_state_path"] = str(state_path)
        updates["issues"] = {**context.get("issues", {}), **output["context"].get("issues", {})}
        return {**output["result"], "status": "success" if output["result"]["status"] in {"complete", "needs_input"} else output["result"]["status"], "updates": updates}

    def generate(context, *, dataset_id):
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
        child["templates_path"] = context["dataset_templates"][dataset_id]
        outcome = run_generation(child, generation)
        child.update(outcome["updates"])
        generations = {**context.get("dataset_generations", {}), dataset_id: {
            "raw_output_path": child["raw_output_path"], "working_blueprint_path": child["working_blueprint_path"]}}
        return result("success", "Dataset generated", dataset_generations=generations, artifact_versions=child["artifact_versions"])

    def finalize(context):
        """Delegate global finalization only after all positive-allocation Generations are complete."""
        if "finalize" not in available_stages(context):
            raise ValueError("All Generations must complete before finalization")
        saved = context.get("finalization_state_path")
        child = json.loads(Path(saved).read_text(encoding="utf-8")) if saved else {}
        child.update({k: context[k] for k in (*COMMON, "selected_dataset_ids", "dataset_assets", "dataset_task_plans",
                     "dataset_templates", "dataset_generations", "allocation_path", "operation_directory") if k in context})
        inject_answers(context, child, "FinalizationAgent")
        output = finalization.invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        path = Path(context["operation_directory"]) / "finalization-state.json"
        atomic_write_json(path, output["context"])
        updates = {"artifact_versions": output["context"]["artifact_versions"], "finalization_state_path": str(path),
                   "last_agent_result": output["result"]}
        updates["issues"] = {**context.get("issues", {}), **output["context"].get("issues", {})}
        updates["global_issue"] = output["context"].get("global_issue")
        if output["result"]["status"] == "complete":
            updates.update({k: v for k, v in output["context"].items() if k in BenchmarkOutputState.__annotations__})
        return {**output["result"], "status": "success" if output["result"]["status"] in {"complete", "needs_input"} else output["result"]["status"], "updates": updates}

    def finish(context):
        if context["run_intent"]["intent"] == "prepare_dataset":
            if preparation_pending(context):
                raise ValueError("Prepare and register the dataset first")
            return result("complete", "Dataset prepared and registered")
        require(context, "Delivery")
        delivery = restore_delivery(context)
        return result("complete", f"Benchmark delivered; shortfall {delivery['total_shortfall']}", **delivery)

    supervisor = compile_agent("SupervisorAgent",
        "Coordinate the user's medical dataset preparation or benchmark construction. "
        "Set run intent from the request when absent; update it when the user adds preparation targets. "
        "Observe list_datasets and inspect_dataset_status before preparation. Source paths come only from config; "
        "never ask the user for paths or invent dataset IDs. Default to reuse_or_prepare; rebuild only when explicitly requested. "
        "Respect the entry intent. Prepare each outstanding logical dataset independently, never merge datasets. "
        "Preparation-only work finishes after all requested registrations; no benchmark brief is needed. "
        "For construction, extract benchmark-level scope, prepare requested new data, then select prepared datasets "
        "with high recall. Plan each selected dataset then prepare_templates; return template task defects to Planning. "
        "Allocate only when every Plan and template set is current. Generate every positive allocation, then delegate "
        "global finalize. Finish after delivery publication. Follow artifact readiness, not a fixed script. "
        "Delegate explicit objectives, constraints, preferences and revision notes to Planning and Allocation. "
        "Delegations must respect the original user requirement; never invent a task focus. "
        "Unsupported plans and zero allocation are valid, failures are not. Never omit explicitly requested datasets. "
        "Clarify global_issue through ask_user with its exact issue_id. After receiving an answer call resolve_clarification: "
        "semantic answers explain source meaning without changing the brief; requirement_change is only for an explicit "
        "change or completion of benchmark requirements. If ambiguous, ask a follow-up with the same issue_id before classifying. "
        "Re-extract the brief only after requirement_change. Never ask routine confirmations. "
        "Finite exhaustion may finish with reported deficits; no automatic refill loop.",
        {"set_intent": set_intent, "resolve_clarification": resolve_clarification, "specify": specify, "select_datasets": select_datasets,
         "list_datasets": list_datasets, "inspect_dataset_status": inspect_dataset_status,
         "prepare_data": prepare_data, "plan": plan, "prepare_templates": prepare_templates, "allocate": allocate, "generate": generate, "finalize": finalize},
        describe, finish, available_tools=available_stages,
        args_schemas={"resolve_clarification": ResolveClarificationArguments, "set_intent": RunIntent, "prepare_data": PrepareDataArguments,
                      "list_datasets": PageArguments, "inspect_dataset_status": DatasetStatusArguments,
                      "plan": PlanningArguments, "allocate": DelegationArguments,
                      "generate": DatasetStatusArguments, "prepare_templates": DatasetStatusArguments})

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
