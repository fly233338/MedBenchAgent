"""Interactive LangGraph orchestration for benchmark planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from medbenchagent.benchmark_info import extract_benchmark_info, missing_fields, write_working_blueprint
from medbenchagent.contracts import GenerationRequest
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
from medbenchagent.generation import (
    advance_generation,
    finalize_generation,
    generation_done,
    initialize_generation,
)
from medbenchagent.llm import complete_json
from medbenchagent.pdf_rag import LocalPdfRag, design_task, discover_medical_materials, write_task_design
from medbenchagent.sample_counts import distribute_sample_counts, write_sample_counts
from medbenchagent.task_screening import (
    apply_task_plan,
    generate_task_plan,
    revise_task_plan,
)

PROJECT_ROOT = Path(__file__).parents[3]
MATERIALS_ROOT = PROJECT_ROOT / "materials"


class BenchmarkInputState(TypedDict, total=False):
    """Values accepted when starting or resuming the root graph."""

    messages: list[Any]
    user_requirement: str
    seed: int
    max_tasks_per_image: int
    question_temperature: float


class BenchmarkState(MessagesState, total=False):
    """Only compact summaries and artifact paths are checkpointed, never table rows."""

    benchmark_info: dict[str, Any]
    user_requirement: str
    benchmark_feedback: str
    benchmark_confirmed: bool
    working_blueprint_path: str
    dataset_recommendation: dict[str, str]
    selected_dataset_id: str
    dataset_confirmed: bool
    dataset_profile: dict[str, Any]
    dataset_access_manifest: dict[str, Any]
    draft_task_plan: dict[str, Any]
    pending_task_feedback: dict[str, str] | None
    task_plan_confirmed: bool
    task_screening_report_path: str
    draft_sample_counts: dict[str, int]
    sample_counts_confirmed: bool
    selected_material_files: list[str]
    material_selection_error: str
    generation_request: GenerationRequest
    raw_output_path: str
    generation_shortfalls: dict[str, int]
    item_count: int
    task_counts: dict[str, int]


class BenchmarkOutputState(TypedDict):
    """Stable result returned by the root benchmark graph."""

    benchmark_info: dict[str, Any]
    selected_dataset_id: str
    working_blueprint_path: str
    task_screening_report_path: str
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


class GenerationInputState(TypedDict):
    """The fixed planning snapshot accepted by the generation subgraph."""

    generation_request: GenerationRequest


class GenerationGraphState(TypedDict, total=False):
    """Private, checkpointed state for the record-by-record generation loop."""

    generation_request: GenerationRequest
    generation_target_counts: dict[str, int]
    generation_completed_counts: dict[str, int]
    generation_cursor: int
    generation_progress_path: str
    generation_rejections: dict[str, int]
    generation_exhausted: bool
    generation_shortfalls: dict[str, int]
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]


class GenerationOutputState(TypedDict):
    """Stable result returned from the generation subgraph to the root graph."""

    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]
    generation_shortfalls: dict[str, int]


class RagDesignInputState(TypedDict):
    """The fixed inputs accepted by the private RAG-design subgraph."""

    benchmark_info: dict[str, Any]
    working_blueprint_path: str
    selected_material_files: list[str]


class RagDesignState(RagDesignInputState, total=False):
    """Private, checkpointed state for the RAG design and review loop."""

    rag_task_queue: list[str]
    rag_task_feedback: dict[str, str]
    rag_current_task_id: str
    rag_completed_count: int
    rag_total_count: int
    designs_confirmed: bool


class RagDesignOutputState(TypedDict):
    """Stable result returned from the RAG-design subgraph to the root graph."""

    working_blueprint_path: str


def _emit_custom_event(event: dict[str, Any]) -> None:
    """Publish live progress when the current invocation is streamed."""
    try:
        get_stream_writer()(event)
    except RuntimeError:
        # Direct node calls remain useful in unit tests and local diagnostics.
        return


def _latest_user_text(state: BenchmarkState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) or getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _benchmark_extract_node(state: BenchmarkState) -> dict[str, Any]:
    """Compute BenchmarkInfo; user confirmation is deliberately a separate node."""
    requirement = state.get("user_requirement") or _latest_user_text(state)
    if not requirement:
        raise ValueError("A benchmark requirement is required before BenchmarkInfo extraction.")
    if feedback := state.get("benchmark_feedback"):
        requirement = f"{requirement}\nRequested correction: {feedback}"
    info = extract_benchmark_info(requirement, state.get("benchmark_info", {}))
    if missing := missing_fields(info):
        raise ValueError(f"BenchmarkInfo is missing required fields: {', '.join(missing)}")
    return {"user_requirement": requirement, "benchmark_info": info, "benchmark_confirmed": False}


def _benchmark_confirm_node(state: BenchmarkState) -> dict[str, Any]:
    """Interpret a natural-language confirmation or correction for BenchmarkInfo."""
    answer = interrupt(
        {
            "kind": "benchmark_info_confirmation",
            "benchmark_info": state["benchmark_info"],
            "question": "Confirm this BenchmarkInfo, or describe the correction.",
        }
    )
    action = _benchmark_review_action(answer, state["benchmark_info"])
    if action["action"] == "confirm":
        working_blueprint_path = state.get("working_blueprint_path") or write_working_blueprint(
            state["benchmark_info"], PROJECT_ROOT / "benchmarks" / "runs"
        )
        return {
            "benchmark_confirmed": True,
            "benchmark_feedback": "",
            "working_blueprint_path": working_blueprint_path,
        }
    return {"benchmark_confirmed": False, "benchmark_feedback": action["feedback"]}


def _benchmark_review_action(answer: object, benchmark_info: dict[str, Any]) -> dict[str, str]:
    action = complete_json(
        "Interpret a BenchmarkInfo review response. Return JSON only and do not change values yourself.",
        "Convert the user's response into exactly one of these shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","feedback":"the requested correction"}\n'
        "Confirm only when the user accepts the complete BenchmarkInfo. For revise, preserve every requested "
        "correction in feedback.\n"
        f"Current BenchmarkInfo: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"User response: {answer}",
    )
    if action == {"action": "confirm"}:
        return {"action": "confirm"}
    if (
        set(action) == {"action", "feedback"}
        and action.get("action") == "revise"
        and isinstance(action.get("feedback"), str)
        and action["feedback"].strip()
    ):
        return {"action": "revise", "feedback": action["feedback"].strip()}
    raise ValueError("BenchmarkInfo review response could not be mapped to one valid confirm or revise action.")


def _dataset_recommend_node(state: BenchmarkState) -> dict[str, Any]:
    datasets = load_dataset_catalog()
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
        '{"action":"select","dataset_id":"one exact available dataset id"}\n'
        "Confirm only when the user accepts the current recommendation. Select only when the user "
        "explicitly requests another available dataset.\n"
        f"Dataset catalog: {json.dumps(public_dataset_entries(datasets), ensure_ascii=False)}\n"
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


def _dataset_review_node(state: BenchmarkState) -> dict[str, Any]:
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
    action = _dataset_review_action(answer, datasets)
    if action["action"] == "confirm":
        dataset = dataset_by_id(datasets, recommendation["dataset_id"])
        return {"selected_dataset_id": dataset["id"], "dataset_confirmed": True}
    dataset = dataset_by_id(datasets, action["dataset_id"])
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


def _dataset_profile_node(state: BenchmarkState) -> dict[str, Any]:
    dataset_id = state.get("selected_dataset_id")
    if not dataset_id:
        raise DatasetCatalogError("A confirmed dataset is required before profiling.")
    dataset = dataset_by_id(load_dataset_catalog(), dataset_id)
    annotation_path = resolve_annotation_path(dataset)
    try:
        profile, manifest = prepare_annotation_dataset(
            annotation_path, PROJECT_ROOT / "benchmarks" / "dataset_cache"
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


def _task_plan_generate_node(state: BenchmarkState) -> dict[str, Any]:
    """Generate all decisions once; this model node never interrupts or edits Blueprint."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    return {
        "draft_task_plan": generate_task_plan(blueprint, state["dataset_profile"]),
        "pending_task_feedback": None,
        "task_plan_confirmed": False,
    }


def _review_action(answer: object, task_ids: list[str]) -> dict[str, str]:
    action = complete_json(
        "Interpret a Task Plan review response. Return JSON only. Do not make task decisions yourself.",
        "Convert the user's response into exactly one of these shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","task_id":"one exact available task id","feedback":"the requested change"}\n'
        "Confirm only when the user accepts the complete Task Plan. A revise action may target only one task. "
        "Preserve the user's requested change in feedback. "
        f"Available task ids: {json.dumps(task_ids, ensure_ascii=False)}\n"
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


def _task_plan_review_node(state: BenchmarkState) -> dict[str, Any]:
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
    action = _review_action(answer, task_ids)
    if action["action"] == "confirm":
        return {"task_plan_confirmed": True, "pending_task_feedback": None}
    return {"task_plan_confirmed": False, "pending_task_feedback": action}


def _task_plan_revise_node(state: BenchmarkState) -> dict[str, Any]:
    """Replace one reviewed entry; all other draft entries are retained verbatim."""
    feedback = state.get("pending_task_feedback")
    if not feedback:
        raise ValueError("Task Plan revision requires pending task feedback.")
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    return {
        "draft_task_plan": revise_task_plan(
            blueprint,
            state["dataset_profile"],
            state["draft_task_plan"],
            feedback["task_id"],
            feedback["feedback"],
        ),
        "pending_task_feedback": None,
        "task_plan_confirmed": False,
    }


def _apply_task_plan_node(state: BenchmarkState) -> dict[str, Any]:
    """Mutate the working Blueprint only after explicit Task Plan confirmation."""
    report_path = apply_task_plan(
        state["working_blueprint_path"], state["draft_task_plan"], state["dataset_profile"]
    )
    return {
        "task_screening_report_path": report_path,
    }


def _sample_count_prepare_node(state: BenchmarkState) -> dict[str, Any]:
    """Prepare a stable even allocation without mutating the working Blueprint."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    total = blueprint.get("benchmark_info", {}).get("sample_count")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint must contain retained tasks for sample allocation.")
    return {
        "draft_sample_counts": distribute_sample_counts(list(tasks), total),
        "sample_counts_confirmed": False,
    }


def _validate_sample_count_action(
    response: object, task_ids: list[str]
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    task_ids = list(current_counts)
    prompt = (
        "Convert the user's sample-count review into exactly one JSON shape:\n"
        '{"action":"confirm"}\n'
        '{"action":"adjust","counts":{"exact task id":0}}\n'
        "For adjust, include only tasks the user explicitly changes. Convert relative requests such as "
        "'add two diagnosis questions' into the final integer for that task using the current allocation. "
        "Do not distribute unspecified counts yourself. Counts must be non-negative integers.\n"
        f"Benchmark total: {total}\n"
        f"Current allocation: {json.dumps(current_counts, ensure_ascii=False)}\n"
        f"User response: {answer}"
    )
    response = complete_json("Interpret a benchmark sample-count review. Return JSON only.", prompt)
    try:
        return _validate_sample_count_action(response, task_ids)
    except ValueError as first_error:
        repaired = complete_json(
            "Repair a benchmark sample-count review action. Return JSON only.",
            f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
            f"Contract error: {first_error}",
        )
        return _validate_sample_count_action(repaired, task_ids)


def _sample_count_review_node(state: BenchmarkState) -> dict[str, Any]:
    counts = state["draft_sample_counts"]
    total = int(state["benchmark_info"]["sample_count"])
    answer = interrupt(
        {
            "kind": "sample_count_confirmation",
            "sample_counts": counts,
            "total": total,
            "question": "Confirm the per-task sample counts or describe adjustments.",
        }
    )
    action = _sample_count_review_action(answer, counts, total)
    if action["action"] == "confirm":
        write_sample_counts(state["working_blueprint_path"], counts)
        return {"sample_counts_confirmed": True}
    return {
        "draft_sample_counts": distribute_sample_counts(
            list(counts), total, action["counts"]
        ),
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
        'Convert the user request into exactly {"selected_files":["exact relative PDF path"]}.\n'
        "Select one or more files only from the scanned medical references. Use exact paths from the list, "
        "including their category directory. Do not select directories or invent filenames.\n"
        f"Scanned medical references: {json.dumps(materials, ensure_ascii=False)}\n"
        f"User response: {answer}"
    )
    response = complete_json("Interpret a medical-reference selection. Return JSON only.", prompt)
    try:
        return _validate_material_selection(response, materials)
    except ValueError as first_error:
        repaired = complete_json(
            "Repair a medical-reference selection. Return JSON only.",
            f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
            f"Contract error: {first_error}",
        )
        return _validate_material_selection(repaired, materials)


def _materials_choose_node(state: BenchmarkState) -> dict[str, Any]:
    materials = discover_medical_materials(MATERIALS_ROOT)
    answer = interrupt(
        {
            "kind": "materials_choose",
            "materials": materials,
            "error": state.get("material_selection_error", ""),
            "question": "Choose one or more medical PDF references by filename or describe the references you need.",
        }
    )
    try:
        selected_files = _material_selection_action(answer, materials)
    except ValueError as exc:
        return {"selected_material_files": [], "material_selection_error": str(exc)}
    return {"selected_material_files": selected_files, "material_selection_error": ""}


def _rag_prepare_node(state: RagDesignState) -> dict[str, Any]:
    """Initialize the ordered RAG queue from the current working Blueprint."""
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint must retain at least one task before RAG design.")
    queue = [task_id for task_id, task in tasks.items() if task.get("sample_count", 0) > 0]
    if not queue:
        raise ValueError("Working Blueprint must allocate samples to at least one task.")
    return {
        "rag_task_queue": queue,
        "rag_task_feedback": {},
        "rag_current_task_id": "",
        "rag_completed_count": 0,
        "rag_total_count": len(queue),
        "designs_confirmed": False,
    }


def _rag_design_task_node(state: RagDesignState) -> dict[str, Any]:
    """Design and persist exactly one Blueprint task."""
    queue = state.get("rag_task_queue", [])
    if not queue:
        raise ValueError("RAG task queue is empty.")
    task_id = queue[0]
    blueprint_path = Path(state["working_blueprint_path"])
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    task = blueprint.get("tasks", {}).get(task_id)
    if not isinstance(task, dict):
        raise ValueError(f"Working Blueprint does not contain queued task: {task_id}")
    rag = LocalPdfRag(
        MATERIALS_ROOT,
        PROJECT_ROOT / "benchmarks" / "rag_cache",
        state["selected_material_files"],
    )
    design = design_task(
        rag,
        blueprint.get("benchmark_info", state["benchmark_info"]),
        task_id,
        task,
        feedback=state.get("rag_task_feedback", {}).get(task_id, ""),
    )
    write_task_design(blueprint_path, task_id, design)
    remaining = queue[1:]
    completed = state.get("rag_completed_count", 0) + 1
    _emit_custom_event(
        {
            "kind": "rag_progress",
            "task_id": task_id,
            "completed": completed,
            "total": state.get("rag_total_count", len(queue)),
            "remaining": len(remaining),
        }
    )
    return {
        "rag_task_queue": remaining,
        "rag_current_task_id": task_id,
        "rag_completed_count": completed,
        "designs_confirmed": False,
    }


def _validate_design_review_action(response: object, task_ids: list[str]) -> dict[str, Any]:
    if response == {"action": "confirm"}:
        return {"action": "confirm"}
    if not isinstance(response, dict) or set(response) != {"action", "revisions"}:
        raise ValueError("Design review must confirm or provide revisions.")
    revisions = response.get("revisions")
    if response.get("action") != "revise" or not isinstance(revisions, list) or not revisions:
        raise ValueError("Design revisions must be a non-empty list.")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for revision in revisions:
        if not isinstance(revision, dict) or set(revision) != {"task_id", "feedback"}:
            raise ValueError("Each design revision must contain task_id and feedback.")
        task_id = revision.get("task_id")
        feedback = revision.get("feedback")
        if task_id not in task_ids or task_id in seen:
            raise ValueError("Design revisions must target unique current Blueprint tasks.")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("Design revision feedback must be non-empty.")
        seen.add(task_id)
        normalized.append({"task_id": task_id, "feedback": feedback.strip()})
    return {"action": "revise", "revisions": normalized}


def _design_review_action(answer: object, tasks: dict[str, Any]) -> dict[str, Any]:
    task_ids = list(tasks)
    prompt = (
        "Convert the user's review into exactly one of these JSON shapes:\n"
        '{"action":"confirm"}\n'
        '{"action":"revise","revisions":[{"task_id":"exact task id","feedback":"requested design change"}]}\n'
        "Confirm only when the complete Blueprint task design is accepted. Support multiple task revisions "
        "and expand a global request to every affected task. Revisions may change only question_templates, "
        "options, generation_logic, or medical_task_description. They must not add or remove tasks or alter "
        "supporting_fields, task_type, or response_format.\n"
        f"Current tasks: {json.dumps(tasks, ensure_ascii=False)}\nUser response: {answer}"
    )
    response = complete_json("Interpret a Question Design review response. Return JSON only.", prompt)
    try:
        return _validate_design_review_action(response, task_ids)
    except ValueError as first_error:
        repaired = complete_json(
            "Repair a Question Design review action. Return JSON only.",
            f"{prompt}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
            f"Contract error: {first_error}",
        )
        return _validate_design_review_action(repaired, task_ids)


def _design_review_node(state: RagDesignState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = {
        task_id: task
        for task_id, task in blueprint.get("tasks", {}).items()
        if task.get("sample_count", 0) > 0
    }
    answer = interrupt(
        {
            "kind": "question_design_confirmation",
            "tasks": tasks,
            "question": "Confirm the Blueprint task designs or describe one or more corrections.",
        }
    )
    action = _design_review_action(answer, tasks)
    if action["action"] == "confirm":
        return {
            "designs_confirmed": True,
            "rag_task_queue": [],
            "rag_task_feedback": {},
        }
    feedback = {revision["task_id"]: revision["feedback"] for revision in action["revisions"]}
    return {
        "designs_confirmed": False,
        "rag_task_queue": list(feedback),
        "rag_task_feedback": feedback,
        "rag_current_task_id": "",
        "rag_completed_count": 0,
        "rag_total_count": len(feedback),
    }


def _generation_request_node(state: BenchmarkState) -> dict[str, Any]:
    blueprint = json.loads(Path(state["working_blueprint_path"]).read_text(encoding="utf-8"))
    tasks = blueprint.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Working Blueprint contains no tasks for generation.")
    benchmark_info = blueprint.get("benchmark_info", state["benchmark_info"])
    request: GenerationRequest = {
        "benchmark_info": benchmark_info,
        "dataset_access_manifest": state["dataset_access_manifest"],
        "tasks": tasks,
        "target_sample_count": int(benchmark_info["sample_count"]),
        "seed": int(state.get("seed", 0)),
        "max_tasks_per_image": int(state.get("max_tasks_per_image", 3)),
        "question_temperature": float(state.get("question_temperature", 0.7)),
        "output_directory": str(Path(state["working_blueprint_path"]).parent),
    }
    return {"generation_request": request}


def _generation_initialize_node(state: GenerationGraphState) -> dict[str, Any]:
    return initialize_generation(state["generation_request"])


def _generation_step_node(state: GenerationGraphState) -> dict[str, Any]:
    update = advance_generation(state["generation_request"], state)
    completed_counts = update.get("generation_completed_counts", state["generation_completed_counts"])
    _emit_custom_event(
        {
            "kind": "generation_progress",
            "cursor": update.get("generation_cursor", state["generation_cursor"]),
            "completed_counts": completed_counts,
            "target_counts": state["generation_target_counts"],
            "rejections": update.get("generation_rejections", state["generation_rejections"]),
            "exhausted": update.get("generation_exhausted", state["generation_exhausted"]),
        }
    )
    return update


def _generation_finalize_node(state: GenerationGraphState) -> dict[str, Any]:
    result = finalize_generation(state["generation_request"], state)
    targets = state["generation_target_counts"]
    shortfalls = {
        task_id: max(0, target - result.task_counts.get(task_id, 0))
        for task_id, target in targets.items()
    }
    return {
        "raw_output_path": result.raw_output_path,
        "generation_completed_counts": result.task_counts,
        "generation_shortfalls": shortfalls,
        "item_count": result.item_count,
        "task_counts": result.task_counts,
    }


def compile_rag_designs_subgraph(*, checkpointer: Any = None):
    """Build the checkpointable RAG design and review loop."""
    graph = StateGraph(
        RagDesignState,
        input_schema=RagDesignInputState,
        output_schema=RagDesignOutputState,
    )
    graph.add_node("rag_prepare", _rag_prepare_node)
    graph.add_node("rag_design_task", _rag_design_task_node)
    graph.add_node("design_review", _design_review_node)
    graph.add_edge(START, "rag_prepare")
    graph.add_edge("rag_prepare", "rag_design_task")
    graph.add_conditional_edges(
        "rag_design_task",
        lambda state: "rag_design_task" if state.get("rag_task_queue") else "design_review",
        {"rag_design_task": "rag_design_task", "design_review": "design_review"},
    )
    graph.add_conditional_edges(
        "design_review",
        lambda state: END if state.get("designs_confirmed") else "rag_design_task",
        {END: END, "rag_design_task": "rag_design_task"},
    )
    return graph.compile(checkpointer=checkpointer, name="rag_designs")


def _rag_designs_node(state: BenchmarkState, config: RunnableConfig) -> dict[str, Any]:
    """Run RAG as one expandable root stage while preserving child checkpoints."""
    compile_rag_designs_subgraph().invoke(
        {
            "benchmark_info": state["benchmark_info"],
            "working_blueprint_path": state["working_blueprint_path"],
            "selected_material_files": state["selected_material_files"],
        },
        config,
    )
    return {}


def compile_generation_subgraph(*, checkpointer: Any = None):
    """Build the checkpointable record-by-record generation loop."""
    graph = StateGraph(
        GenerationGraphState,
        input_schema=GenerationInputState,
        output_schema=GenerationOutputState,
    )
    graph.add_node("generation_initialize", _generation_initialize_node)
    graph.add_node("generation_step", _generation_step_node)
    graph.add_node("generation_finalize", _generation_finalize_node)
    graph.add_edge(START, "generation_initialize")
    graph.add_conditional_edges(
        "generation_initialize",
        lambda state: "generation_finalize" if generation_done(state) else "generation_step",
        {
            "generation_step": "generation_step",
            "generation_finalize": "generation_finalize",
        },
    )
    graph.add_conditional_edges(
        "generation_step",
        lambda state: "generation_finalize" if generation_done(state) else "generation_step",
        {
            "generation_step": "generation_step",
            "generation_finalize": "generation_finalize",
        },
    )
    graph.add_edge("generation_finalize", END)
    return graph.compile(checkpointer=checkpointer, name="generate_raw_benchmark")


def _generate_raw_benchmark_node(state: BenchmarkState, config: RunnableConfig) -> dict[str, Any]:
    """Run generation as one expandable root stage while preserving child checkpoints."""
    return compile_generation_subgraph().invoke(
        {"generation_request": state["generation_request"]}, config
    )


def compile_benchmark_graph(*, checkpointer: Any = None):
    """Build the interactive planning and benchmark generation workflow."""
    graph = StateGraph(
        BenchmarkState,
        input_schema=BenchmarkInputState,
        output_schema=BenchmarkOutputState,
    )
    graph.add_node("benchmark_extract", _benchmark_extract_node)
    graph.add_node("benchmark_confirm", _benchmark_confirm_node)
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
    graph.add_node("rag_designs", _rag_designs_node)
    graph.add_node("generation_request", _generation_request_node)
    graph.add_node("generate_raw_benchmark", _generate_raw_benchmark_node)
    graph.add_edge(START, "benchmark_extract")
    graph.add_edge("benchmark_extract", "benchmark_confirm")
    graph.add_conditional_edges(
        "benchmark_confirm",
        lambda state: "dataset_recommend" if state.get("benchmark_confirmed") else "benchmark_extract",
        {"dataset_recommend": "dataset_recommend", "benchmark_extract": "benchmark_extract"},
    )
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
        lambda state: "rag_designs" if state.get("selected_material_files") else "materials_choose",
        {"rag_designs": "rag_designs", "materials_choose": "materials_choose"},
    )
    graph.add_edge("rag_designs", "generation_request")
    graph.add_edge("generation_request", "generate_raw_benchmark")
    graph.add_edge("generate_raw_benchmark", END)
    return graph.compile(checkpointer=checkpointer)


benchmark_graph = compile_benchmark_graph()
