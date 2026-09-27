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
from medbenchagent.runtime.contexts import SupervisorContext
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.tool_contracts import PrepareDataArguments, DatasetId, DelegationArguments, RunIntent, PageArguments, NoArguments
from medbenchagent.data.readiness import list_datasets, inspect_dataset_status, dataset_status, configured_source, reuse_prepared
from medbenchagent.runtime.run_artifacts import initialize_run, atomic_write_json
from medbenchagent.settings import get_settings, run_settings
from medbenchagent.runtime.contexts import load_context, save_context, ROLES
from medbenchagent.runtime.dependencies import publish, require, initialize_dependencies, verify_resume
from medbenchagent.runtime.preflight import preflight_errors, rag_capability
from medbenchagent.runtime.clarifications import inject_answers, new_issue, resolve_clarification, ResolveClarificationArguments


class DatasetStatusArguments(NoArguments):
    dataset_id: DatasetId


class PlanningArguments(DelegationArguments):
    dataset_id: DatasetId


class BenchmarkInputState(TypedDict, total=False):
    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    max_total_tasks_per_image: int | None
    question_temperature: float
    prepare_dataset_id: str
    intake_dataset_name: str
    intake_source_root: str
    requested_rebuild: bool
    entry_intent: str
    requested_dataset_ids: list[str]
    experiment_name: str
    experiments_root: str


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


def specify(context: SupervisorContext):
    """Extract the benchmark brief from the user's complete requirement."""
    if context.get("brief_requirement") == context["user_requirement"]:
        raise ValueError("This requirement has already been extracted; clarify missing fields first")
    user_inputs = context.get("user_inputs", [context["user_requirement"]])
    info = extract_benchmark_info(context["user_requirement"], user_inputs=user_inputs)
    evidence = info.pop("evidence")
    preferences = info.pop("allocation_preferences")
    sample_scope = info.pop("sample_scope", [])
    total_limit = info.pop("max_total_tasks_per_image", None)
    if info.get("modalities") and not any(r["kind"] == "modality" for r in sample_scope):
        raise ValueError("Explicit modalities must be represented in sample_scope; re-extract the requirement")
    source = "user"
    if info.get("sample_count") is None:
        info["sample_count"] = get_settings().benchmark.default_sample_count
        source = "configuration"
    missing = missing_fields(info)
    updates = {"benchmark_info": info, "sample_scope": sample_scope, "requirement_evidence": evidence,
               "allocation_preferences": preferences,
               "user_inputs": user_inputs,
               "max_total_tasks_per_image": total_limit if total_limit is not None else context.get("configured_total_tasks_per_image"),
               "brief_ready": not missing, "brief_missing_fields": missing,
               "brief_requirement": context["user_requirement"], "sample_count_source": source}
    if missing:
        questions = {"clinical_scope": "请说明要评测的医学范围或数据集范围。",
                     "language": "请指定 benchmark 输出语言：英文（en）或中文（zh）。",
                     "sample_count": "请指定正整数题量。"}
        question = "\n".join(questions[field] for field in missing)
        issue = new_issue("SupervisorAgent", context, context["user_requirement"], question)
        issue.update(purpose="benchmark_fields", missing_fields=missing)
        updates.update(global_issue=issue, issues={**context.get("issues", {}), issue["issue_id"]: issue})
        return result("success", "Brief needs clarification: " + ", ".join(missing), **updates)
    path = Path(context["operation_directory"]) / "blueprint.json"
    write_working_blueprint(info, path)
    publish(context, "GenerationConfig", {"seed": context.get("seed", get_settings().generation.seed),
            "max_tasks_per_image": context.get("max_tasks_per_image", get_settings().generation.max_tasks_per_image),
            "max_total_tasks_per_image": updates["max_total_tasks_per_image"]}, ())
    return result("success", "Benchmark brief extracted", **updates,
                  working_blueprint_path=str(path),
                  **publish(context, "Requirement", {"path": str(path), "brief": info,
                             "user_requirement": context["user_requirement"], "sample_count_source": source,
                             "evidence": evidence, "allocation_preferences": preferences,
                             "sample_scope": sample_scope, "max_total_tasks_per_image": updates["max_total_tasks_per_image"]}, ()))


COMMON = ("run_id", "run_directory", "construction_events_path", "effective_config_path", "artifact_versions",
          "limits", "user_requirement", "benchmark_info", "seed", "max_tasks_per_image", "question_temperature",
          "registration_catalog_path", "sample_scope", "max_total_tasks_per_image", "allocation_preferences")


def required_field_action(context):
    """Route confirmed intake and code-created field clarifications without a model decision."""
    issue = context.get("global_issue")
    if issue and issue.get("purpose") == "benchmark_fields":
        return {"tool": "ask_user", "arguments": {"question": issue["question"], "issue_id": issue["issue_id"]},
                "reason": "Complete required benchmark fields"}
    if (context.get("entry_intent") == "prepare_dataset" and context.get("intake_dataset_name")
            and context.get("run_intent", {}).get("intent") == "prepare_dataset" and not issue):
        pending = preparation_pending(context)
        if not pending:
            return {"tool": "finish", "arguments": {}, "reason": "Preparation registration is complete"}
        if context.get("last_agent_result", {}).get("status") not in {"failed", "blocked"}:
            return {"tool": "prepare_data", "arguments": {
                "dataset_id": pending[0],
                "mode": "rebuild" if context.get("requested_rebuild") else "reuse_or_prepare",
            }, "reason": "Prepare the dataset confirmed during intake"}
    clarification = context.get("clarification") or {}
    if (clarification.get("purpose") == "benchmark_fields"
            and context.get("brief_requirement") != context["user_requirement"]):
        return {"tool": "specify", "arguments": {}, "reason": "Re-extract completed benchmark requirement"}
    return None


def canonical_delegation(context, dataset=None):
    """Stable requirement references, not Supervisor paraphrases, define delegated work."""
    requirement = require(context, "Requirement")["Requirement"]
    return {"objective": "Follow the requested benchmark scope", "dataset_id": dataset,
            "requirement_version": requirement["id"],
            "scope_ids": [r["id"] for r in context.get("sample_scope", [])],
            "preference_ids": [p["id"] for p in context.get("allocation_preferences", [])]}


def agent_operation(context, agent, dataset_id=None, role=None):
    root = Path(context["run_directory"]) / "agents"
    parent = Path(context["operation_directory"]).relative_to(root / "SupervisorAgent")
    path = root / agent
    if dataset_id:
        path /= dataset_id
    if role:
        path /= role
    return str(path / parent)


def child_context(context, dataset_id, role):
    if dataset_id not in context.get("selected_dataset_ids", []):
        raise ValueError("Dataset is not selected")
    require(context, "DatasetSelection")
    saved = context.get("child_states", {}).get(dataset_id, {}).get(role)
    child = load_context(saved, role, dataset_id) if saved else {}
    child.update({k: context[k] for k in COMMON if k in context})
    asset = context["dataset_assets"][dataset_id]
    child.update(agent_role=ROLES[role], dataset_scope=dataset_id, selected_dataset_id=dataset_id,
                 annotation_path=asset["access"]["annotation_path"], dataset_entry=asset["entry"],
                 dataset_access_manifest=asset["access"], dataset_card_summary=asset["card_summary"],
                 operation_directory=agent_operation(
                     context, {"planning": "TaskPlanningAgent", "templates": "TaskPlanningAgent",
                               "generation": "GenerationAgent"}[role], dataset_id, role))
    return inject_answers(context, child, {"planning": "TaskPlanningAgent", "generation": "GenerationAgent",
                                          "templates": "TaskPlanningAgent"}[role], dataset_id)


def child_result(context, dataset_id, role, output):
    child = output["context"]
    path = Path(child["operation_directory"]) / "state.json"
    save_context(path, child, role, dataset_id)
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


def describe(context: SupervisorContext):
    details = {key: context.get(key) for key in (
        "user_requirement", "benchmark_info", "brief_ready", "brief_missing_fields", "sample_count_source", "prepare_dataset_id", "selected_dataset_ids",
        "intake_dataset_name", "intake_source_root", "requested_rebuild",
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


def available_stages(context: SupervisorContext):
    if context.get("global_issue"):
        return {"resolve_clarification", "ask_user"} if context["global_issue"]["status"] == "answered" else {"ask_user"}
    tools = {"ask_user", "list_datasets", "inspect_dataset_status"}
    intent = context.get("run_intent")
    if not intent:
        return tools | {"set_intent"}
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


SUPERVISOR_INSTRUCTIONS = """Coordinate the user's medical dataset preparation or benchmark construction.

Role and authority:
The session has already selected entry_intent. Preserve it; set_intent is needed only when run_intent is absent.
Preparation and construction are separate runs. Do not infer new tasks or benchmark priorities from delegation notes.
Use the confirmed dataset ID and source inputs; inspect dataset status when readiness or reuse is uncertain.
An intake_source_root is already confirmed; do not ask for that path again. SourceDiscovery verifies its contents.
Default to reuse_or_prepare. Rebuild only when explicitly requested. Never merge independent datasets.

Decisions:
Use outstanding_work, current artifact readiness and available tools to choose the next useful action.
A saved path alone does not prove readiness. Preserve completed valid work and never omit explicitly requested datasets.
Construction requires a brief, selection, each local Plan and Templates, global Allocation, positive-quota Generation,
and global Finalization. Resolve template task defects in Planning. Code supplies delegation objectives; revision_notes
are diagnostic only. A child's complete means its local work completed, not that the benchmark was delivered.
needs_input requires the existing clarification; blocked/failed is not an unsupported task or permission to skip a dataset.
Re-delegate only with changed inputs or a concrete correction, never to reset a failed budget.

Clarification and recovery:
Use global_issue and its exact issue_id. Reuse existing answers rather than asking again.
Missing benchmark fields are handled automatically. For other answered issues, resolve_clarification classifies semantic
explanations versus an explicit change/completion of requirements. Only the latter updates the brief.
If ambiguous, ask a focused follow-up with the same issue_id. Do not ask routine confirmations or invent missing semantics.

Completion:
Preparation finishes after all requested registrations; it needs no benchmark brief.
Construction finishes only after validated delivery publication. Unsupported plans and zero allocations are legitimate
outcomes, execution failures are not. Finite generation exhaustion reports shortfall; do not start an automatic refill loop."""


def compile_benchmark_graph(*, checkpointer=None):
    planning = compile_task_planning_agent()
    generation = compile_generation_agent()
    allocation = compile_allocation_agent()
    finalization = compile_finalization_agent()

    def select_datasets(context):
        """Select only datasets prepared when this build run began."""
        return run_selection(context)

    def plan(context, *, dataset_id, delegation=None):
        """Plan tasks for exactly one selected prepared dataset; no quotas are assigned here."""
        require_pending(context, "plan", dataset_id)
        child = child_context(context, dataset_id, "planning")
        child["delegation"] = canonical_delegation(context, dataset_id)
        child["diagnostic_notes"] = (delegation or {}).get("revision_notes", "")
        publish(child, "PlanningDelegation", child["delegation"], ("Requirement",))
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
        if context.get("requested_rebuild"):
            mode = "rebuild"
        elif mode == "rebuild" and context.get("requested_rebuild") is False:
            raise ValueError("Rebuild requires an explicit user request")
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
        configured = configured_source(dataset_id)
        discovered = context.get("discovered_sources", {}).get(dataset_id)
        if configured is None and discovered is None and status["available"] is None:
            from medbenchagent.agents.source_discovery import discover_source
            discovered = discover_source(context, dataset_id)
        if discovered is None and status["source_errors"]:
            return result("blocked", "; ".join(status["source_errors"]))
        sources = (discovered or configured)["sources"]
        if saved:
            from medbenchagent.data.contracts import Source
            child = load_context(saved, "preparation", dataset_id)
            if child["sources"] != [Source.model_validate(s).model_dump() for s in sources]:
                raise ValueError("Preparation sources changed; start a new experiment")
            if child["preparation_mode"] != mode:
                raise ValueError("Preparation mode changed; resume with the original mode")
            child.update({k: context[k] for k in COMMON if k in context})
            child["operation_directory"] = agent_operation(context, "DataProcessingAgent", dataset_id)
        else:
            child = {k: context[k] for k in COMMON if k in context}
            child["operation_directory"] = agent_operation(context, "DataProcessingAgent", dataset_id)
            child = initialize(child, sources, dataset_id)
            child["preparation_mode"] = mode
        child["feedback"] = feedback
        inject_answers(context, child, "DataProcessingAgent", dataset_id)
        output = compile_data_processing_agent().invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "preparation", output)
        if discovered is not None:
            outcome["updates"]["discovered_sources"] = {**context.get("discovered_sources", {}),
                                                            dataset_id: discovered}
            outcome["updates"]["source_decision_path"] = str(
                Path(context["run_directory"]) / "discovery" / "sources" /
                dataset_id / "source_decision.json")
        if output["result"]["status"] == "complete":
            entries = {**context.get("prepared_entries", {}), dataset_id: output["context"]["dataset_entry"]}
            outcome["updates"]["prepared_entries"] = entries
            published = dict(context, artifact_versions=outcome["updates"]["artifact_versions"])
            outcome["updates"].update(publish(published, "PreparedCandidates", {"entries": entries}, ()))
        return outcome

    def allocate(context, *, delegation=None):
        """Delegate global Concept-first allocation after every selected dataset plan is complete."""
        saved = context.get("allocation_state_path")
        child = load_context(saved, "allocation") if saved else {}
        child.update({k: context[k] for k in (*COMMON, "selected_dataset_ids", "dataset_task_plans") if k in context})
        child["operation_directory"] = agent_operation(context, "AllocationAgent")
        inject_answers(context, child, "AllocationAgent")
        child["delegation"] = canonical_delegation(context)
        child["diagnostic_notes"] = (delegation or {}).get("revision_notes", "")
        child["allocation_dataset_cards"] = {d: context["dataset_assets"][d]["card_summary"] for d in context["selected_dataset_ids"]}
        publish(child, "AllocationDelegation", child["delegation"], ("Requirement",))
        from medbenchagent.agents.allocation import _plans
        if not any(leaf["eligible_count"] for p in _plans(child).values() for leaf in p["tasks"].values()):
            return result("blocked", "All selected datasets completed planning without eligible tasks")
        output = allocation.invoke({"context": child})
        state_path = Path(child["operation_directory"]) / "state.json"
        save_context(state_path, output["context"], "allocation")
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
        child = load_context(saved, "finalization") if saved else {}
        child.update({k: context[k] for k in (*COMMON, "selected_dataset_ids", "dataset_assets", "dataset_task_plans",
                     "dataset_templates", "dataset_generations", "allocation_path") if k in context})
        child["operation_directory"] = agent_operation(context, "FinalizationAgent")
        inject_answers(context, child, "FinalizationAgent")
        output = finalization.invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        path = Path(child["operation_directory"]) / "state.json"
        save_context(path, output["context"], "finalization")
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
        SUPERVISOR_INSTRUCTIONS,
        {"set_intent": set_intent, "resolve_clarification": resolve_clarification, "specify": specify, "select_datasets": select_datasets,
         "list_datasets": list_datasets, "inspect_dataset_status": inspect_dataset_status,
         "prepare_data": prepare_data, "plan": plan, "prepare_templates": prepare_templates, "allocate": allocate, "generate": generate, "finalize": finalize},
        describe, finish, available_tools=available_stages, automatic_action=required_field_action,
        read_tools=frozenset({"list_datasets", "inspect_dataset_status"}),
        args_schemas={"resolve_clarification": ResolveClarificationArguments, "set_intent": RunIntent, "prepare_data": PrepareDataArguments,
                      "list_datasets": PageArguments, "inspect_dataset_status": DatasetStatusArguments,
                      "plan": PlanningArguments, "allocate": DelegationArguments,
                      "generate": DatasetStatusArguments, "prepare_templates": DatasetStatusArguments})

    def initialize(state):
        overrides = {k: state[k] for k in ("seed", "max_tasks_per_image", "max_total_tasks_per_image", "question_temperature") if k in state}
        requirement = state.get("user_requirement") or next((getattr(m, "content", "") for m in reversed(convert_to_messages(state.get("messages", []))) if getattr(m, "type", "") == "human"), "")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("A natural-language requirement is required")
        settings = get_settings()
        if errors := preflight_errors(settings, build=state.get("entry_intent") == "build_benchmark"):
            raise ValueError("Local preflight failed: " + "; ".join(errors))
        from medbenchagent.runtime.model_services import resolve_run_services

        settings = resolve_run_services(settings)
        defaults = {k: getattr(settings.generation, k) for k in ("seed", "max_tasks_per_image", "max_total_tasks_per_image", "question_temperature")}
        run = initialize_run(settings=settings, graph_overrides=overrides)
        context = dict(run, **(defaults | overrides), user_requirement=requirement,
                       operation_directory=str(Path(run["run_directory"]) / "agents" /
                                               "SupervisorAgent"), limits=settings.harness,
                       registration_catalog_path=str(settings.paths.dataset_catalog),
                       rag_capability=rag_capability(settings),
                       prepare_dataset_id=state.get("prepare_dataset_id"),
                       intake_dataset_name=state.get("intake_dataset_name"),
                       intake_source_root=state.get("intake_source_root"),
                       requested_rebuild=state.get("requested_rebuild"),
                       entry_intent=state.get("entry_intent"),
                       requested_dataset_ids=state.get("requested_dataset_ids"))
        context["experiment_name"] = state.get("experiment_name")
        context["experiments_root"] = state.get("experiments_root")
        context["configured_total_tasks_per_image"] = context["max_total_tasks_per_image"]
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
            from medbenchagent.runtime.model_services import verify_frozen_services

            verify_frozen_services(get_settings())
            if errors := preflight_errors(get_settings(), build=state["context"].get("run_intent", {}).get("intent") == "build_benchmark"):
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
