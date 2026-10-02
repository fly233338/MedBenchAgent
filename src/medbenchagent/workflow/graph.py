"""Supervisor orchestration with explicit dataset-local child state and global allocation."""

import json
import re
from pathlib import Path
from typing import Any, TypedDict
from langchain_core.messages import convert_to_messages
from langgraph.graph import StateGraph, START, END
from medbenchagent.agents.generation import run_generation
from medbenchagent.agents.finalization import restore_delivery
from medbenchagent.agents.templates import prepare_templates as run_templates
from medbenchagent.agents.task_planning import compile_task_planning_agent, apply_template_feedback
from medbenchagent.agents.v2_pipeline import run_task_space_pipeline
from medbenchagent.planning.dataset_selection import discover_dataset_candidates as run_discovery
from medbenchagent.planning.dataset_selection import select_datasets as run_selection
from medbenchagent.planning.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.runtime.contexts import SupervisorContext
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.llm import StructuredOutputError
from medbenchagent.runtime.tool_contracts import PrepareDataArguments, DatasetId, DelegationArguments, PageArguments, NoArguments, NonEmpty
from pydantic import Field
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


class DatasetSelectionArguments(NoArguments):
    dataset_ids: list[DatasetId] = Field(min_length=1)
    selection_reason: dict[str, NonEmpty] = Field(min_length=1)
    selection_issue_id: str = ""


class BenchmarkInputState(TypedDict, total=False):
    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    max_total_tasks_per_image: int | None
    question_temperature: float
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
    dataset_candidates: list[dict]
    candidate_order: list[str]
    preparation_candidates: list[str]
    candidate_snapshot_path: str
    selection_path: str
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
    dataset_candidates: list[dict]
    candidate_order: list[str]
    preparation_candidates: list[str]
    candidate_snapshot_path: str
    selection_path: str
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
    if errors := preflight_errors(get_settings(), build=True):
        return result("blocked", "Benchmark preflight failed: " + "; ".join(errors))
    if context.get("brief_requirement") == context["user_requirement"]:
        raise ValueError("This requirement has already been extracted; clarify missing fields first")
    user_inputs = context.get("user_inputs", [context["user_requirement"]])
    try:
        info = extract_benchmark_info(context["user_requirement"], user_inputs=user_inputs)
    except StructuredOutputError as exc:
        issue = new_issue("SupervisorAgent", context, context["user_requirement"],
                          "I could not reliably extract the benchmark requirements from your last message. "
                          "Please restate the benchmark scope, requested output language, and any dataset names "
                          "or sample constraints in one concise sentence.")
        issue.update(purpose="benchmark_fields", missing_fields=["extraction"], diagnostic=str(exc))
        updates = {"brief_ready": False, "brief_missing_fields": ["extraction"], "global_issue": issue,
                   "issues": {**context.get("issues", {}), issue["issue_id"]: issue}}
        return result("success", "Benchmark brief needs a clearer requirement", **updates)
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
        questions = {"clinical_scope": "Please describe the medical scope or population that the benchmark should evaluate.",
                     "language": "Which language should the generated benchmark use? Reply with English (en) or Chinese (zh).",
                     "sample_count": "Please provide a positive integer for the requested sample count."}
        question = "\n".join(questions[field] for field in missing)
        issue = new_issue("SupervisorAgent", context, context["user_requirement"], question)
        issue.update(purpose="benchmark_fields", missing_fields=missing)
        updates.update(global_issue=issue, issues={**context.get("issues", {}), issue["issue_id"]: issue})
        return result("success", "Brief needs clarification: " + ", ".join(missing), **updates)
    path = Path(context["operation_directory"]) / "benchmark_brief.json"
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
          "registration_catalog_path", "sample_scope", "max_total_tasks_per_image", "allocation_preferences",
          "execution_runtime")


def required_field_action(context):
    """Route code-created benchmark field clarifications without a model decision."""
    issue = context.get("global_issue")
    if issue and issue.get("purpose") == "benchmark_fields":
        return {"tool": "ask_user", "arguments": {"question": issue["question"], "issue_id": issue["issue_id"]},
                "reason": "Complete required benchmark fields"}
    if issue and issue.get("purpose") == "dataset_selection" and issue.get("status") == "pending":
        return {"tool": "ask_user", "arguments": {"question": issue["question"], "issue_id": issue["issue_id"]},
                "reason": "Collect the user's dataset selection"}
    clarification = context.get("clarification") or {}
    if (clarification.get("purpose") == "benchmark_fields"
            and context.get("brief_requirement") != context["user_requirement"]):
        return {"tool": "specify", "arguments": {}, "reason": "Re-extract completed benchmark requirement"}
    return None


def automatic_supervisor_action(context):
    """Route only code-created benchmark-field clarifications automatically."""
    return required_field_action(context)


def canonical_delegation(context, dataset=None):
    """Stable requirement references, not Supervisor paraphrases, define delegated work."""
    requirement = require(context, "Requirement")["Requirement"]
    return {"objective": "Follow the requested benchmark scope", "dataset_id": dataset,
            "requirement_version": requirement["id"],
            "scope_ids": [r["id"] for r in context.get("sample_scope", [])],
            "preference_ids": [p["id"] for p in context.get("allocation_preferences", [])]}


def agent_operation(context, agent, dataset_id=None, role=None):
    root = Path(context["run_directory"]) / "agents"
    path = root / agent
    if dataset_id:
        path /= dataset_id
    if role:
        path /= role
    return str(path)


def child_context(context, dataset_id, role):
    if dataset_id not in context.get("selected_dataset_ids", []):
        raise ValueError("Dataset is not selected")
    require(context, "DatasetSelection")
    saved = context.get("child_states", {}).get(dataset_id, {}).get(role)
    child = load_context(saved, role, dataset_id) if saved else {}
    child.update({k: context[k] for k in COMMON if k in context})
    asset = context["dataset_assets"][dataset_id]
    child.update(agent_role=ROLES[role], dataset_scope=dataset_id, selected_dataset_id=dataset_id,
                 cases_path=asset["access"]["cases_path"], dataset_entry=asset["entry"],
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
        "user_requirement", "benchmark_info", "brief_ready", "brief_missing_fields", "sample_count_source", "selected_dataset_ids",
        "dataset_task_plans", "dataset_templates", "template_revision_requests", "dataset_generations", "delivery_counts", "allocation_path", "prepared_entries",
        "artifact_versions", "global_issue", "last_agent_result", "requested_dataset_ids",
        "dataset_candidates", "candidate_order", "preparation_candidates", "candidate_snapshot_path")}
    ids = set(context.get("selected_dataset_ids", [])) | set(context.get("prepared_entries", {}))
    details["dataset_readiness"] = {d: {kind: _current(context, f"dataset/{d}/{kind}")
                                      for kind in ("Dataset", "Plan", "Templates", "Generation")} for d in sorted(ids)}
    details["outstanding_work"] = {**pending_datasets(context),
                                   "available_actions": sorted(available_stages(context))}
    details["rag_capability"] = context.get("rag_capability")
    return details


def _current(context, *kinds):
    try:
        require(context, *kinds)
        return True
    except ValueError:
        return False


def preparation_pending(context):
    return [d for d in context.get("preparation_required", [])
            if d not in context.get("prepared_entries", {}) or not _current(context, f"dataset/{d}/Dataset")]


def pending_datasets(context):
    pending = {"pending_plan_dataset_ids": [], "pending_templates_dataset_ids": [], "pending_generation_dataset_ids": []}
    if not _current(context, "DatasetSelection"):
        return pending
    for dataset in context["selected_dataset_ids"]:
        feedback = context.get("template_revision_requests", {}).get(dataset)
        if not _current(context, f"dataset/{dataset}/Plan") or (feedback and feedback["plan_version"] == context["artifact_versions"].get(f"dataset/{dataset}/Plan")):
            pending["pending_plan_dataset_ids"].append(dataset)
        elif context.get("dataset_assets", {}).get(dataset, {}).get("access", {}).get("schema_version") == "annotation.v5":
            continue
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


def _construction_started(context: SupervisorContext) -> bool:
    versions = context.get("artifact_versions", {})
    return any(key in versions for key in ("Requirement", "DatasetSelection", "Allocation", "Delivery"))


def _rebuild_requested(context: SupervisorContext) -> bool:
    """Require an explicit user request before discarding a prepared version."""
    texts = [context.get("user_requirement", ""), *context.get("user_inputs", [])]
    pattern = re.compile(r"(?:\brebuild\b|\bforce(?:d)?\s+rebuild\b|\bredo\b|重新(?:构建|准备|生成)|强制重做)", re.IGNORECASE)
    return any(isinstance(text, str) and pattern.search(text) for text in texts)


def available_stages(context: SupervisorContext):
    if context.get("global_issue"):
        issue = context["global_issue"]
        if issue.get("purpose") == "dataset_selection" and issue.get("status") == "answered":
            return {"ask_user", "read_evidence", "prepare_data", "discover_datasets", "select_datasets"}
        return {"resolve_clarification", "ask_user", "read_evidence"} if issue["status"] == "answered" else {"ask_user"}
    tools = {"ask_user", "list_datasets", "inspect_dataset_status", "prepare_data", "discover_datasets"}
    if not _construction_started(context):
        tools.add("end")
    if context.get("prepared_entries") and "Requirement" not in context.get("artifact_versions", {}):
        tools.add("finish")
    if context.get("brief_requirement") != context.get("user_requirement") or "brief_requirement" not in context:
        tools.add("specify")
    if not context.get("brief_ready"):
        return tools
    if not _current(context, "DatasetCandidates"):
        return tools | {"discover_datasets"}
    if not _current(context, "DatasetSelection"):
        return tools | {"select_datasets"}
    v5_selected = [d for d in context.get("selected_dataset_ids", [])
                   if context.get("dataset_assets", {}).get(d, {}).get("access", {}).get("schema_version") == "annotation.v5"]
    if v5_selected:
        if pending_datasets(context)["pending_plan_dataset_ids"]:
            return tools | {"plan"}
        if _current(context, "Delivery"):
            return tools | {"finish"}
        return tools | {"run_v2"}
    if any("planning" in state for state in context.get("child_states", {}).values()):
        tools.add("run_v2")
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


SUPERVISOR_INSTRUCTIONS = """You coordinate a resumable medical VLM dataset-preparation and benchmark-construction run.

Mission and boundaries:
- Stay within raw dataset preparation, prepared-dataset selection, benchmark planning, generation, and delivery.
- Use the current state and tool results as facts. Never invent dataset metadata, labels, paths, or readiness.
- Keep semantic interpretation in the model; let tools enforce IDs, schemas, digests, dependencies, and side effects.
- For unrelated requests, call end without creating construction artifacts.

Intake and preparation:
- Read the complete user request and clarification history before choosing an action.
- Call specify when the benchmark brief is missing or stale. It extracts explicit intent and preserves source evidence.
- If preparation is requested, use a readable raw path from the request or configured source. Normalize a path-derived ID
  only when needed and call prepare_data with reuse_or_prepare; use rebuild only for an explicit rebuild request.
- Ask one focused English question for a missing or unreadable raw path. Do not ask the user to describe files that the
  DataProcessingAgent can inspect itself.

Candidate discovery and selection:
- After the brief is ready, call discover_datasets once. It freezes validated Prepared candidates and returns bounded
  Card summaries; it never chooses or publishes a dataset selection.
- If the user explicitly named one or more available candidates, map the names to frozen IDs and call select_datasets.
- If no dataset was explicitly named, use the benchmark intent and Card summaries to identify a high-recall relevant set,
  then ask the user in English to choose one or more candidates. Include stable numbers, dataset IDs, names, scope,
  modality, and case count. Pass purpose=dataset_selection, the candidate snapshot path, and allowed IDs to ask_user.
- A selection answer may use names, numbers, IDs, or natural language. Read the answer evidence when needed, map it to
  frozen IDs, and call select_datasets with one or more IDs and a reason for each. The tool is the final authority.
- If the answer is ambiguous, ask a concise English follow-up with the same issue_id. Never select an ID outside the snapshot.
- If a requested dataset is not prepared but has a configured source, prepare it before rediscovering candidates. If no
  source exists, ask for the raw directory or let the user choose another candidate.

Handoff and construction:
- Do not call plan, run_v2, allocation, generation, or finalization before DatasetSelection is published.
- Delegate planning using the canonical Requirement and selected dataset references. Do not paraphrase or discard scope.
- Preserve completed valid work. A child completion is local progress; the run is complete only after validated delivery.
- Resolve existing issues by their exact issue_id. Do not repeat an unchanged failed action or start an automatic refill loop.
- Unsupported plans and zero allocations are valid outcomes; execution failures require diagnosis or user input.

Interaction:
- All Supervisor-generated questions must be in English, even when the user's input is Chinese. The benchmark output
  language is a separate requirement and may be English or Chinese according to the brief.
- Ask only for information that changes the next safe action. Use existing answers and evidence before asking again."""


def compile_benchmark_graph(*, checkpointer=None):
    planning = compile_task_planning_agent()
    generation = None
    finalization = None

    def discover_datasets(context):
        """Freeze and expose validated Prepared candidates for Supervisor review."""
        return run_discovery(context)

    def select_datasets(context, *, dataset_ids, selection_reason, selection_issue_id=""):
        """Bind one or more user/Supervisor-selected candidates from the frozen snapshot."""
        return run_selection(context, dataset_ids=dataset_ids, selection_reason=selection_reason,
                             selection_issue_id=selection_issue_id)

    def plan(context, *, dataset_id, delegation=None):
        """Plan tasks for exactly one selected prepared dataset; no quotas are assigned here."""
        require_pending(context, "plan", dataset_id)
        child = child_context(context, dataset_id, "planning")
        child["delegation"] = canonical_delegation(context, dataset_id)
        child["diagnostic_notes"] = (delegation or {}).get("revision_notes", "")
        # Task Planning is a resumable data-driven loop. Give it a budget
        # proportional to the Node space instead of the generic 60-call
        # supervisor budget; every mutation is checkpointed in task_space_plan.
        base_limits = context.get("limits", {})
        child["limits"] = {
            **base_limits,
            "max_calls": max(base_limits.get("max_calls", 60), 1200),
            "max_seconds": max(base_limits.get("max_seconds", 3600), 14400),
        }
        publish(child, "PlanningDelegation", child["delegation"], ("Requirement",))
        feedback = context.get("template_revision_requests", {}).get(dataset_id)
        if feedback:
            correction = apply_template_feedback(child, feedback)
            if correction["status"] == "blocked":
                return correction
            child.update(correction["updates"])
        output = planning.invoke({"context": child}, config={"recursion_limit": 3 * context.get("limits", {}).get("max_calls", 60) + 10})
        outcome = child_result(context, dataset_id, "planning", output)
        # A long planning loop may hit the per-invocation harness budget after
        # checkpointing a valid partial draft. Keep it resumable instead of
        # treating a budget boundary as a semantic failure.
        draft = output.get("context", {}).get("task_space_plan", {})
        if output["result"]["status"] == "blocked" and draft.get("phase") not in {"complete", "blocked"}:
            outcome["status"] = "success"
            outcome["summary"] = "Task Plan progress saved; planning can resume from the current Node"
            outcome["updates"]["planning_incomplete"] = True
        if output["result"]["status"] == "complete":
            requests = dict(context.get("template_revision_requests", {}))
            requests.pop(dataset_id, None)
            outcome["updates"]["template_revision_requests"] = requests
            outcome["updates"]["dataset_task_plans"] = {**context.get("dataset_task_plans", {}), dataset_id: output["context"]["working_blueprint_path"]}
        return outcome

    def run_v2(context, *, dataset_id):
        """Execute a completed v2 Task Space for one prepared Dataset."""
        saved = context.get("child_states", {}).get(dataset_id, {}).get("planning")
        if not saved:
            raise ValueError("Run v2 planning tools before executing the Task Space pipeline")
        child = load_context(saved, "planning", dataset_id)
        child["operation_directory"] = agent_operation(context, "TaskSpacePipelineAgent", dataset_id)
        output = run_task_space_pipeline(child)
        atomic_write_json(Path(child["operation_directory"]) / "result.json", output)
        result_path = Path(child["operation_directory"]) / "result.json"
        results = {**context.get("v2_pipeline_results", {}), dataset_id: str(result_path)}
        counts = {**context.get("v2_pipeline_counts", {}), dataset_id: {"accepted": output["audit"]["accepted"], "shortfall": output.get("shortfall", {})}}
        updates = {"v2_pipeline_results": results, "v2_pipeline_counts": counts,
                   "v2_pipeline_path": str(result_path), "delivery_counts": counts}
        if set(results) >= set(context.get("selected_dataset_ids", [])):
            versions = publish(context, "Delivery", {"task_space_outputs": results}, ())
            updates.update(versions, delivery_path=str(result_path))
        return result("success", output["summary"], dataset_id=dataset_id, **updates)

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
        child.update(outcome.get("updates", {}))
        state_path = Path(child["operation_directory"]) / "state.json"
        save_context(state_path, child, "templates", dataset_id)
        states = {key: dict(value) for key, value in context.get("child_states", {}).items()}
        states.setdefault(dataset_id, {})["templates"] = str(state_path)
        outcome["updates"]["child_states"] = states
        return outcome

    def prepare_data(context, *, dataset_id, mode="reuse_or_prepare", feedback=""):
        """Reuse valid prepared data or prepare configured sources. Use rebuild only for an explicit user request."""
        from medbenchagent.runtime.preparation_context import initialize_preparation
        from medbenchagent.agents.data_processing import compile_data_processing_agent
        if mode == "rebuild" and not _rebuild_requested(context):
            return result("failed", "Rebuild requires an explicit user request; use reuse_or_prepare or ask the user.")
        operation = Path(context["operation_directory"])
        if mode == "reuse_or_prepare" and (operation / "reused_dataset.json").is_file():
            outcome = reuse_prepared(context, dataset_id)
            outcome.setdefault("updates", {})["preparation_required"] = [
                item for item in context.get("preparation_required", []) if item != dataset_id
            ]
            return outcome
        status = dataset_status(context, dataset_id)
        saved = context.get("child_states", {}).get(dataset_id, {}).get("preparation")
        if not saved and not (operation / "preparation.json").is_file() and mode == "reuse_or_prepare":
            if status["prepared"]:
                outcome = reuse_prepared(context, dataset_id)
                outcome.setdefault("updates", {})["preparation_required"] = [
                    item for item in context.get("preparation_required", []) if item != dataset_id
                ]
                return outcome
            if status["available"] is not None:
                return result("blocked", "Existing dataset needs repair or explicit rebuild: " + "; ".join(status["issues"]))
        configured = configured_source(dataset_id)
        discovered = context.get("discovered_sources", {}).get(dataset_id)
        if configured is None and discovered is None and status["available"] is None:
            from medbenchagent.runtime.source_discovery import discover_source
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
            child = initialize_preparation(child, sources, dataset_id)
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
            # Coding Agent submissions are finalized deterministically by the
            # host before they are exposed to planning.
            if "dataset_entry" not in output["context"]:
                from medbenchagent.data.finalization import finalize_prepared_dataset
                from medbenchagent.runtime.construction import record_event
                candidate = output["context"].get("preparation_candidate", {})
                candidate_path = candidate.get("candidate_path") if isinstance(candidate, dict) else None
                if candidate_path:
                    event_path = output["context"].get("construction_events_path")
                    event_base = {"run_id": output["context"].get("run_id"),
                                  "agent": "DatasetProcessingAgent", "dataset_id": dataset_id,
                                  "workspace": output["context"].get("preparation_workspace"),
                                  "conversation_id": output["context"].get("preparation_conversation_id"),
                                  "candidate_path": candidate_path}
                    if event_path:
                        record_event(event_path, "finalization_started", operation="finalize_prepared_dataset",
                                     status="running", **event_base)
                    try:
                        finalized = finalize_prepared_dataset(candidate_path, output["context"])
                    except (ValueError, OSError) as exc:
                        if event_path:
                            record_event(event_path, "finalization_finished", operation="finalize_prepared_dataset",
                                         status="blocked", diagnostic=str(exc), **event_base)
                        # Validation feedback is persisted for the same Coding
                        # Session; the next Supervisor turn performs continuation.
                        return {"status": "blocked", "summary": str(exc),
                                "updates": {**outcome.get("updates", {}),
                                            "preparation_status": "blocked",
                                            "validation_error": str(exc),
                                            "preparation_feedback": str(exc)}}
                    if event_path:
                        record_event(event_path, "finalization_finished", operation="finalize_prepared_dataset",
                                     status="complete", dataset_entry=finalized.get("dataset_entry"),
                                     validation_report=finalized.get("validation_report"), **event_base)
                    from medbenchagent.data.dataset_catalog import register_dataset
                    registered_entry = register_dataset(
                        dataset_id, finalized["bundle"], catalog_path=context["registration_catalog_path"]
                    )
                    finalized["dataset_entry"] = registered_entry
                    output["context"].update({"dataset_entry": registered_entry,
                                               "cases_path": finalized["cases_path"],
                                               "dataset_access_manifest": finalized["access_manifest"]})
                    dataset_versions = publish(
                        output["context"], "Dataset",
                        {"cases": finalized["cases_path"], "access": finalized["access_manifest"]},
                        ("DataConfig",),
                    )
                    profile_versions = publish(
                        output["context"], "DatasetProfile",
                        {"path": finalized["access_manifest"].get("profile_path")}, ("Dataset",),
                    )
                    card_versions = {} if finalized.get("annotation_schema") == "annotation.v5" else publish(
                        output["context"], "DatasetCard",
                        {"path": finalized["card_path"]}, ("DatasetProfile",),
                    )
                    outcome["updates"].update({
                        "dataset_entry": finalized["dataset_entry"],
                        "cases_path": finalized["cases_path"],
                        "dataset_access_manifest": finalized["access_manifest"],
                        "artifact_versions": {
                            **outcome["updates"].get("artifact_versions", {}),
                            **dataset_versions.get("artifact_versions", {}),
                            **profile_versions.get("artifact_versions", {}),
                            **card_versions.get("artifact_versions", {}),
                        },
                    })
            entries = {**context.get("prepared_entries", {}), dataset_id: output["context"]["dataset_entry"]}
            outcome["updates"]["prepared_entries"] = entries
            outcome["updates"]["preparation_required"] = [
                item for item in context.get("preparation_required", []) if item != dataset_id
            ]
            published = dict(context, artifact_versions=outcome["updates"]["artifact_versions"])
            outcome["updates"].update(publish(published, "PreparedCandidates", {"entries": entries}, ()))
        return outcome

    def allocate(context, *, delegation=None):
        """Legacy allocation is unavailable; v5 uses the deterministic Task Space pipeline."""
        raise ValueError("Use run_v2 for annotation.v5 datasets")

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
        state_path = Path(child["operation_directory"]) / "state.json"
        save_context(state_path, child, "generation", dataset_id)
        states = {key: dict(value) for key, value in context.get("child_states", {}).items()}
        states.setdefault(dataset_id, {})["generation"] = str(state_path)
        generations = {**context.get("dataset_generations", {}), dataset_id: {
            "raw_output_path": child["raw_output_path"], "working_blueprint_path": child["working_blueprint_path"]}}
        return result("success", "Dataset generated", dataset_generations=generations,
                      artifact_versions=child["artifact_versions"], child_states=states)

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
        if context.get("delivery_path") and any(
            context.get("dataset_assets", {}).get(d, {}).get("access", {}).get("schema_version") == "annotation.v5"
            for d in context.get("selected_dataset_ids", [])
        ):
            return result("complete", f"Task Space benchmark delivered; shortfall {context.get('delivery_counts', {}).get('shortfall', {})}",
                          delivery_path=context["delivery_path"], delivery_counts=context.get("delivery_counts", {}))
        if context.get("prepared_entries") and "Requirement" not in context.get("artifact_versions", {}):
            if preparation_pending(context):
                raise ValueError("Prepare and register the dataset first")
            return result("complete", "Dataset prepared and registered")
        require(context, "Delivery")
        delivery = restore_delivery(context)
        return result("complete", f"Benchmark delivered; shortfall {delivery['total_shortfall']}", **delivery)

    def end(context):
        if _construction_started(context):
            return result("blocked", "This benchmark request has started construction and cannot be ended as out of scope.")
        return result("complete", "This request is outside MedBenchAgent's scope")

    supervisor = compile_agent("SupervisorAgent",
        SUPERVISOR_INSTRUCTIONS,
        {"end": end, "resolve_clarification": resolve_clarification, "specify": specify,
         "discover_datasets": discover_datasets, "select_datasets": select_datasets,
         "list_datasets": list_datasets, "inspect_dataset_status": inspect_dataset_status,
         "prepare_data": prepare_data, "plan": plan, "run_v2": run_v2, "prepare_templates": prepare_templates, "allocate": allocate, "generate": generate, "finalize": finalize},
        describe, finish, available_tools=available_stages, automatic_action=automatic_supervisor_action,
        read_tools=frozenset({"list_datasets", "inspect_dataset_status", "discover_datasets"}),
        args_schemas={"resolve_clarification": ResolveClarificationArguments, "prepare_data": PrepareDataArguments,
                      "list_datasets": PageArguments, "inspect_dataset_status": DatasetStatusArguments,
                      "select_datasets": DatasetSelectionArguments,
                      "plan": PlanningArguments, "run_v2": DatasetStatusArguments, "allocate": DelegationArguments,
                      "generate": DatasetStatusArguments, "prepare_templates": DatasetStatusArguments})

    def initialize(state):
        overrides = {k: state[k] for k in ("seed", "max_tasks_per_image", "max_total_tasks_per_image", "question_temperature") if k in state}
        requirement = state.get("user_requirement") or next((getattr(m, "content", "") for m in reversed(convert_to_messages(state.get("messages", []))) if getattr(m, "type", "") == "human"), "")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("A natural-language requirement is required")
        settings = get_settings()
        if errors := preflight_errors(settings):
            raise ValueError("Local preflight failed: " + "; ".join(errors))
        from medbenchagent.runtime.model_services import resolve_run_services
        from medbenchagent.data.execution import select_backend

        settings = resolve_run_services(settings)
        execution_runtime = select_backend(settings.data_processing)
        defaults = {k: getattr(settings.generation, k) for k in ("seed", "max_tasks_per_image", "max_total_tasks_per_image", "question_temperature")}
        run = initialize_run(settings=settings, graph_overrides=overrides,
                             execution_runtime=execution_runtime)
        context = dict(run, **(defaults | overrides), user_requirement=requirement,
                       execution_runtime=execution_runtime,
                       operation_directory=str(Path(run["run_directory"]) / "agents" /
                                               "SupervisorAgent"), limits=settings.harness,
                       registration_catalog_path=str(settings.paths.dataset_catalog),
                       rag_capability=rag_capability(settings))
        context["experiment_name"] = state.get("experiment_name")
        context["experiments_root"] = state.get("experiments_root")
        context["configured_total_tasks_per_image"] = context["max_total_tasks_per_image"]
        initialize_dependencies(context, settings)
        publish(context, "PreparedCandidates", {"entries": {}}, ())
        return dict(run, context=context)

    def run_supervisor(state):
        verify_resume(state["context"])
        with run_settings(state["context"]["effective_config_path"]):
            from medbenchagent.runtime.model_services import verify_frozen_services

            verify_frozen_services(get_settings())
            if errors := preflight_errors(get_settings()):
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
