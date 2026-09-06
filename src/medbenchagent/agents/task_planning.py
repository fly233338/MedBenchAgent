"""TaskPlanningAgent: dataset, task-plan, allocation, and material confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from medbenchagent.agents._context import artifact_snapshot, human_interaction, llm_context, revision_applied
from medbenchagent.agents.contracts import (
    TASK_PLANNING_AGENT,
    TaskPlanningAgentInputState,
    TaskPlanningAgentOutputState,
    TaskPlanningAgentState,
)
from medbenchagent.construction import changed_mapping_keys, event_operation
from medbenchagent.dataset import AnnotationValidationError, prepare_annotation_dataset
from medbenchagent.dataset_catalog import (
    DatasetCatalogError,
    DatasetEntry,
    dataset_by_id,
    load_dataset_catalog,
    public_dataset_entries,
    recommend_dataset,
    resolve_annotation_path,
)
from medbenchagent.llm import complete_json
from medbenchagent.rag.pdf import discover_medical_materials
from medbenchagent.run_artifacts import update_blueprint_dataset
from medbenchagent.sample_counts import distribute_weighted_sample_counts, write_sample_counts
from medbenchagent.settings import get_settings
from medbenchagent.task_screening import (
    apply_task_plan,
    generate_task_plan,
    revise_task_plan,
    write_task_plan_report,
)

def _dataset_recommend_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    datasets = load_dataset_catalog()
    with llm_context(state, TASK_PLANNING_AGENT, "dataset_recommend"):
        recommendation = recommend_dataset(state["benchmark_info"], datasets)
    dataset = dataset_by_id(datasets, recommendation["dataset_id"])
    return {
        "dataset_recommendation": {
            **recommendation,
            "name": dataset["name"],
            "description": dataset["description"],
        },
        "selected_dataset_id": "",
        "dataset_confirmed": False,
    }


def _dataset_review_action(answer: object, datasets: list[DatasetEntry]) -> dict[str, str]:
    response = complete_json(
        "Interpret a medical dataset selection review. Return JSON only.",
        "Convert the user's response into exactly one of these shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"select","dataset_id":"one exact available dataset id"}\n\n'
        "Confirm only when the user accepts the current recommendation. Select only when the user "
        "explicitly requests another available dataset.\n\n"
        f"Dataset catalog: {json.dumps(public_dataset_entries(datasets), ensure_ascii=False)}\n\n"
        f"User response: {answer}",
    )
    if response == {"action": "confirm"}:
        return {"action": "confirm"}
    if (
        isinstance(response, dict)
        and set(response) == {"action", "dataset_id"}
        and response.get("action") == "select"
        and isinstance(response.get("dataset_id"), str)
    ):
        dataset = dataset_by_id(datasets, response["dataset_id"].strip())
        return {"action": "select", "dataset_id": dataset["id"]}
    raise ValueError("Dataset review response could not be mapped to one valid confirm or select action.")


def _dataset_review_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    datasets = load_dataset_catalog()
    recommendation = state["dataset_recommendation"]
    answer = interrupt(
        {
            "kind": "dataset_selection_confirmation",
            "recommendation": recommendation,
            "datasets": public_dataset_entries(datasets),
            "question": "Confirm the recommended dataset, or name another available dataset.",
        }
    )
    try:
        with llm_context(state, TASK_PLANNING_AGENT, "dataset_review"):
            action = _dataset_review_action(answer, datasets)
    except Exception as exc:
        human_interaction(
            state,
            agent=TASK_PLANNING_AGENT,
            stage="dataset",
            user_input=answer,
            parse_status="failed",
            error_type=type(exc).__name__,
        )
        raise
    event_action = "confirm" if action["action"] == "confirm" else "revise"
    interaction_id = human_interaction(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="dataset",
        user_input=answer,
        parse_status="success",
        action=event_action,
    )
    if action["action"] == "confirm":
        dataset = dataset_by_id(datasets, recommendation["dataset_id"])
        update_blueprint_dataset(state["working_blueprint_path"], dataset["id"])
        return {"selected_dataset_id": dataset["id"], "dataset_confirmed": True}
    dataset = dataset_by_id(datasets, action["dataset_id"])
    revision_applied(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="dataset",
        interaction_id=interaction_id,
        modified_object_ids=(
            [dataset["id"]] if dataset["id"] != recommendation["dataset_id"] else []
        ),
    )
    return {
        "dataset_recommendation": {
            "dataset_id": dataset["id"],
            "name": dataset["name"],
            "description": dataset["description"],
            "reason": "Selected by the user during dataset review.",
        },
        "selected_dataset_id": "",
        "dataset_confirmed": False,
    }


def _dataset_profile_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    dataset_id = state.get("selected_dataset_id")
    if not dataset_id:
        raise DatasetCatalogError("A confirmed dataset is required before profiling.")
    dataset = dataset_by_id(load_dataset_catalog(), dataset_id)
    annotation_path = resolve_annotation_path(dataset)
    try:
        profile, manifest = prepare_annotation_dataset(
            annotation_path, get_settings().paths.dataset_cache_root
        )
    except (AnnotationValidationError, ValueError) as exc:
        raise DatasetCatalogError(
            f"Selected dataset {dataset_id} could not be prepared: {exc}"
        ) from exc
    return {
        "dataset_profile": profile,
        "dataset_access_manifest": {
            key: value for key, value in manifest.items() if key != "annotation_path"
        },
        "task_plan_confirmed": False,
    }


def _task_plan_generate_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    """Generate one complete validated Task Plan without interrupting or editing Blueprint."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    with llm_context(state, TASK_PLANNING_AGENT, "task_plan_generate"):
        task_plan = generate_task_plan(blueprint, state["dataset_profile"])
    artifact = write_task_plan_report(state["working_blueprint_path"], task_plan)
    artifact_snapshot(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="task_plan",
        artifact_type="task_plan",
        artifact=artifact,
    )
    return {
        "draft_task_plan": task_plan,
        "task_screening_report_path": str(artifact["path"]),
        "pending_task_feedback": None,
        "task_plan_confirmed": False,
    }


def _review_action(answer: object, task_ids: list[str]) -> dict[str, str]:
    action = complete_json(
        "Interpret a Task Plan review response. Return JSON only. Do not make task decisions yourself.",
        "Convert the user's response into exactly one of these shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","task_id":"one exact available task id","feedback":"the requested change"}\n\n'
        "Confirm only when the user accepts the complete Task Plan. A revise action may target only one task. "
        "Preserve the user's requested change in feedback.\n\n"
        f"Available task ids: {json.dumps(task_ids, ensure_ascii=False)}\n\n"
        f"User response: {answer}",
    )
    if action == {"action": "confirm"}:
        return {"action": "confirm"}
    if (
        set(action) == {"action", "task_id", "feedback"}
        and action.get("action") == "revise"
        and isinstance(action.get("task_id"), str)
        and action["task_id"].strip() in task_ids
        and isinstance(action.get("feedback"), str)
        and action["feedback"].strip()
    ):
        return {
            "action": "revise",
            "task_id": action["task_id"].strip(),
            "feedback": action["feedback"].strip(),
        }
    raise ValueError("Task Plan review response could not be mapped to one valid confirm or revise action.")


def _task_plan_review_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    """Receive one review action, interpreting natural language only when needed."""
    answer = interrupt(
        {
            "kind": "task_plan_review",
            "dataset_profile": state["dataset_profile"],
            "task_plan": state["draft_task_plan"],
            "question": "Confirm the Task Plan, or describe one task that should be revised and how.",
        }
    )
    task_ids = list(state["draft_task_plan"]["tasks"])
    try:
        with llm_context(state, TASK_PLANNING_AGENT, "task_plan_revise"):
            action = _review_action(answer, task_ids)
    except Exception as exc:
        human_interaction(
            state,
            agent=TASK_PLANNING_AGENT,
            stage="task_plan",
            user_input=answer,
            parse_status="failed",
            error_type=type(exc).__name__,
        )
        raise
    interaction_id = human_interaction(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="task_plan",
        user_input=answer,
        parse_status="success",
        action=action["action"],
    )
    if action["action"] == "confirm":
        return {"task_plan_confirmed": True, "pending_task_feedback": None}
    return {
        "task_plan_confirmed": False,
        "pending_task_feedback": action,
        "pending_task_interaction_id": interaction_id,
    }


def _task_plan_revise_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    """Replace one reviewed entry; all other draft entries are retained verbatim."""
    feedback = state.get("pending_task_feedback")
    if not feedback:
        raise ValueError("Task Plan revision requires pending task feedback.")
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    before_tasks = state["draft_task_plan"].get("tasks", {})
    with llm_context(state, TASK_PLANNING_AGENT, "task_plan_revise"):
        revised_task_plan = revise_task_plan(
            blueprint,
            state["dataset_profile"],
            state["draft_task_plan"],
            feedback["task_id"],
            feedback["feedback"],
        )
    after_tasks = revised_task_plan.get("tasks", {})
    artifact = write_task_plan_report(state["working_blueprint_path"], revised_task_plan)
    revision_applied(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="task_plan",
        interaction_id=state.get("pending_task_interaction_id", ""),
        modified_object_ids=changed_mapping_keys(before_tasks, after_tasks),
        artifact_type="task_plan",
        artifact=artifact,
    )
    return {
        "draft_task_plan": revised_task_plan,
        "task_screening_report_path": str(artifact["path"]),
        "pending_task_feedback": None,
        "pending_task_interaction_id": "",
        "task_plan_confirmed": False,
    }


def _apply_task_plan_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    """Mutate the working Blueprint only after explicit Task Plan confirmation."""
    report_path = apply_task_plan(
        state["working_blueprint_path"],
        state["draft_task_plan"],
        state["dataset_profile"],
        state.get("task_screening_report_path"),
    )
    return {"task_screening_report_path": report_path}


def _sample_count_prepare_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    """Prepare a stable category-weighted allocation without mutating the working Blueprint."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    total = blueprint.get("benchmark_info", {}).get("sample_count")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint must contain retained tasks for sample allocation.")
    return {
        "draft_sample_counts": distribute_weighted_sample_counts(tasks, total),
        "sample_counts_confirmed": False,
    }


def _validate_sample_count_action(response: object, task_ids: list[str]) -> dict[str, Any]:
    if response == {"action": "confirm"}:
        return {"action": "confirm"}
    if not isinstance(response, dict) or set(response) != {"action", "counts"}:
        raise ValueError("Sample-count review must confirm or provide counts.")
    counts = response.get("counts")
    if response.get("action") != "adjust" or not isinstance(counts, dict) or not counts:
        raise ValueError("Sample-count adjustments must contain at least one task count.")
    normalized: dict[str, int] = {}
    for task_id, count in counts.items():
        if task_id not in task_ids:
            raise ValueError(f"Unknown task in sample-count adjustment: {task_id}.")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Adjusted sample counts must be non-negative integers.")
        normalized[task_id] = count
    return {"action": "adjust", "counts": normalized}


def _sample_count_review_action(
    answer: object,
    current_counts: dict[str, int],
    total: int,
    tasks: dict[str, Any],
) -> dict[str, Any]:
    task_ids = list(current_counts)
    task_categories = {task_id: tasks[task_id]["category"] for task_id in task_ids}
    active_categories = list(dict.fromkeys(task_categories.values()))
    weights = dict(get_settings().category_weights)
    active_weight_total = sum(weights[category] for category in active_categories)
    normalized_weights = {
        category: weights[category] / active_weight_total
        for category in active_categories
    }
    prompt = (
        "Convert the user's sample-count review into exactly one JSON shape:\n"
        '{"action":"confirm"}\n'
        '{"action":"adjust","counts":{"exact task id":0}}\n\n'
        "For adjust, include only tasks the user explicitly changes. Convert relative requests such as "
        "'add two diagnosis questions' into the final integer for that task using the current allocation. "
        "Do not distribute unspecified counts yourself. Counts must be non-negative integers.\n\n"
        f"Benchmark total: {total}\n\n"
        f"Task categories: {json.dumps(task_categories, ensure_ascii=False)}\n"
        f"Normalized active category weights: {json.dumps(normalized_weights, ensure_ascii=False)}\n"
        f"Current allocation: {json.dumps(current_counts, ensure_ascii=False)}\n\n"
        f"User response: {answer}"
    )
    response = complete_json("Interpret a benchmark sample-count review. Return JSON only.", prompt)
    try:
        return _validate_sample_count_action(response, task_ids)
    except ValueError as first_error:
        with event_operation("repair"):
            repaired = complete_json(
                "Repair a benchmark sample-count review action. Return JSON only.",
                f"{prompt}\n\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n\n"
                f"Contract error: {first_error}",
            )
        return _validate_sample_count_action(repaired, task_ids)


def _sample_count_review_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    counts = state["draft_sample_counts"]
    total = int(state["benchmark_info"]["sample_count"])
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("Working Blueprint must contain tasks for sample-count review.")
    answer = interrupt(
        {
            "kind": "sample_count_confirmation",
            "sample_counts": counts,
            "total": total,
            "question": "Confirm the per-task sample counts or describe adjustments.",
        }
    )
    try:
        with llm_context(state, TASK_PLANNING_AGENT, "sample_count_review"):
            action = _sample_count_review_action(answer, counts, total, tasks)
    except Exception as exc:
        human_interaction(
            state,
            agent=TASK_PLANNING_AGENT,
            stage="sample_count",
            user_input=answer,
            parse_status="failed",
            error_type=type(exc).__name__,
        )
        raise
    event_action = "confirm" if action["action"] == "confirm" else "revise"
    interaction_id = human_interaction(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="sample_count",
        user_input=answer,
        parse_status="success",
        action=event_action,
    )
    if action["action"] == "confirm":
        write_sample_counts(state["working_blueprint_path"], counts)
        return {"sample_counts_confirmed": True}
    revised_counts = distribute_weighted_sample_counts(tasks, total, action["counts"])
    revision_applied(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="sample_count",
        interaction_id=interaction_id,
        modified_object_ids=changed_mapping_keys(counts, revised_counts),
    )
    return {
        "draft_sample_counts": revised_counts,
        "sample_counts_confirmed": False,
    }


def _validate_material_selection(
    response: object, materials: list[dict[str, str]]
) -> list[str]:
    if not isinstance(response, dict) or set(response) != {"selected_files"}:
        raise ValueError('Material selection must be {"selected_files": ["exact relative PDF path"]}.')
    selected_files = response.get("selected_files")
    if not isinstance(selected_files, list) or not selected_files:
        raise ValueError("Select at least one medical PDF.")
    if not all(isinstance(file, str) for file in selected_files):
        raise ValueError("Selected medical PDF paths must be strings.")
    if len(selected_files) != len(set(selected_files)):
        raise ValueError("Selected medical PDF paths must not repeat.")
    available_files = {material["file"] for material in materials}
    unknown_files = [file for file in selected_files if file not in available_files]
    if unknown_files:
        raise ValueError(f"Unknown medical PDF selection: {', '.join(unknown_files)}.")
    return selected_files


def _material_selection_action(answer: object, materials: list[dict[str, str]]) -> list[str]:
    prompt = (
        'Convert the user request into exactly {"selected_files":["exact relative PDF path"]}.\n\n'
        "Select one or more files only from the scanned medical references. Use exact paths from the list, "
        "including their category directory. Do not select directories or invent filenames.\n\n"
        f"Scanned medical references: {json.dumps(materials, ensure_ascii=False)}\n\n"
        f"User response: {answer}"
    )
    response = complete_json("Interpret a medical-reference selection. Return JSON only.", prompt)
    try:
        return _validate_material_selection(response, materials)
    except ValueError as first_error:
        with event_operation("repair"):
            repaired = complete_json(
                "Repair a medical-reference selection. Return JSON only.",
                f"{prompt}\n\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n\n"
                f"Contract error: {first_error}",
            )
        return _validate_material_selection(repaired, materials)


def _materials_choose_node(state: TaskPlanningAgentState) -> dict[str, Any]:
    materials = discover_medical_materials(get_settings().paths.materials_root)
    answer = interrupt(
        {
            "kind": "materials_choose",
            "materials": materials,
            "error": state.get("material_selection_error", ""),
            "question": "Choose one or more medical PDF references by filename or describe the references you need.",
        }
    )
    try:
        with llm_context(state, TASK_PLANNING_AGENT, "material_selection"):
            selected_files = _material_selection_action(answer, materials)
    except ValueError as exc:
        human_interaction(
            state,
            agent=TASK_PLANNING_AGENT,
            stage="materials",
            user_input=answer,
            parse_status="failed",
            error_type=type(exc).__name__,
        )
        return {"selected_material_files": [], "material_selection_error": str(exc)}
    except Exception as exc:
        human_interaction(
            state,
            agent=TASK_PLANNING_AGENT,
            stage="materials",
            user_input=answer,
            parse_status="failed",
            error_type=type(exc).__name__,
        )
        raise
    human_interaction(
        state,
        agent=TASK_PLANNING_AGENT,
        stage="materials",
        user_input=answer,
        parse_status="success",
        action="confirm",
        selected_objects=len(selected_files),
    )
    return {"selected_material_files": selected_files, "material_selection_error": ""}


def compile_task_planning_agent(*, checkpointer: Any = None):
    """Build the checkpointable TaskPlanningAgent planning loop."""
    graph = StateGraph(
        TaskPlanningAgentState,
        input_schema=TaskPlanningAgentInputState,
        output_schema=TaskPlanningAgentOutputState,
    )
    graph.add_node("dataset_recommend", _dataset_recommend_node)
    graph.add_node("dataset_review", _dataset_review_node)
    graph.add_node("dataset_profile", _dataset_profile_node)
    graph.add_node("task_plan_generate", _task_plan_generate_node)
    graph.add_node("task_plan_review", _task_plan_review_node)
    graph.add_node("task_plan_revise", _task_plan_revise_node)
    graph.add_node("apply_task_plan", _apply_task_plan_node)
    graph.add_node("sample_count_prepare", _sample_count_prepare_node)
    graph.add_node("sample_count_review", _sample_count_review_node)
    graph.add_node("materials_choose", _materials_choose_node)
    graph.add_edge(START, "dataset_recommend")
    graph.add_edge("dataset_recommend", "dataset_review")
    graph.add_conditional_edges(
        "dataset_review",
        lambda state: "dataset_profile" if state.get("dataset_confirmed") else "dataset_review",
        {"dataset_profile": "dataset_profile", "dataset_review": "dataset_review"},
    )
    graph.add_conditional_edges(
        "dataset_profile",
        lambda state: "task_plan_generate" if state.get("dataset_profile") else "dataset_profile",
        {"task_plan_generate": "task_plan_generate", "dataset_profile": "dataset_profile"},
    )
    graph.add_edge("task_plan_generate", "task_plan_review")
    graph.add_conditional_edges(
        "task_plan_review",
        lambda state: "apply_task_plan" if state.get("task_plan_confirmed") else "task_plan_revise",
        {"apply_task_plan": "apply_task_plan", "task_plan_revise": "task_plan_revise"},
    )
    graph.add_edge("task_plan_revise", "task_plan_review")
    graph.add_edge("apply_task_plan", "sample_count_prepare")
    graph.add_edge("sample_count_prepare", "sample_count_review")
    graph.add_conditional_edges(
        "sample_count_review",
        lambda state: "materials_choose" if state.get("sample_counts_confirmed") else "sample_count_review",
        {"materials_choose": "materials_choose", "sample_count_review": "sample_count_review"},
    )
    graph.add_conditional_edges(
        "materials_choose",
        lambda state: END if state.get("selected_material_files") else "materials_choose",
        {END: END, "materials_choose": "materials_choose"},
    )
    return graph.compile(checkpointer=checkpointer, name=TASK_PLANNING_AGENT)
