"""Task planning tools reuse the existing planning, allocation and medical design logic."""

from __future__ import annotations
import json
from pathlib import Path
from medbenchagent.dataset import prepare_annotation_dataset
from medbenchagent.image_assets import load_image_manifest
from medbenchagent.contracts import RAG_DESIGN_FIELDS
from medbenchagent.dataset_catalog import (
    DATASET_ID_PATTERN,
    dataset_by_id,
    load_dataset_catalog,
    public_dataset_entries,
    recommend_dataset,
    resolve_annotation_path,
)
from medbenchagent.harness import compile_agent, result
from medbenchagent.tool_contracts import (
    AllocateArguments, MaterialsArguments, MedicalDesignArguments, ProfileArguments,
    ReviseTaskArguments, SelectDatasetArguments,
)
from medbenchagent.rag.pdf import LocalPdfRag, design_task, discover_medical_materials
from medbenchagent.run_artifacts import atomic_write_json, write_versioned_json
from medbenchagent.sample_counts import distribute_weighted_sample_counts
from medbenchagent.settings import get_settings
from medbenchagent.task_screening import apply_task_plan, generate_task_plan, revise_task_plan


def _read(context):
    return json.loads(Path(context["working_blueprint_path"]).read_text(encoding="utf-8"))


def _save(context, blueprint):
    path = Path(context["operation_directory"]) / "blueprint.json"
    atomic_write_json(path, blueprint)
    return str(path)


def select_dataset(context, *, dataset_id: str | None = None, annotation_path: str | None = None):
    """Select standard data; omit both to recommend catalog data."""
    if annotation_path:
        identifier = dataset_id
        if not isinstance(identifier, str) or not DATASET_ID_PATTERN.fullmatch(identifier):
            raise ValueError("Direct standard input requires a lowercase dataset_id")
        path = Path(annotation_path).expanduser().resolve()
        if path.name != "annotation.jsonl" or not path.is_file():
            raise ValueError("Provide an existing standard annotation.jsonl")
        entry = {
            "id": identifier,
            "name": identifier,
            "description": "Standard local dataset",
            "annotation_path": str(path),
            "available": True,
        }
    else:
        catalog = load_dataset_catalog()
        identifier = (
            dataset_id
            or recommend_dataset(context["benchmark_info"], catalog)["dataset_id"]
        )
        entry = dataset_by_id(catalog, identifier)
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    blueprint.update(benchmark_info=context["benchmark_info"], dataset={"dataset_id": identifier})
    return result(
        "success",
        "Dataset selected",
        selected_dataset_id=identifier,
        annotation_path=str(resolve_annotation_path(entry)),
        dataset_entry=entry,
        plan_binding=None,
        dataset_profile=None,
        dataset_access_manifest=None,
        working_blueprint_path=_save(context, blueprint),
    )


def profile_dataset(context, *, revision: str = "1", refresh: bool = False):
    """Validate selected annotations and build profile."""
    if not context.get("annotation_path"):
        raise ValueError("Select or prepare a dataset first")
    load_image_manifest(Path(context["annotation_path"]).parent / "image_manifest.jsonl")
    profile, access = prepare_annotation_dataset(
        context["annotation_path"],
        get_settings().paths.dataset_cache_root,
        revision=revision,
        refresh=refresh,
    )
    return result(
        "success",
        "Dataset profile available",
        dataset_profile=profile,
        dataset_access_manifest=access,
        plan_binding=None,
    )


def plan_tasks(context):
    """Generate the complete task plan using existing logic. Requires a dataset profile."""
    if not context.get("dataset_profile"):
        raise ValueError("Profile the dataset first")
    blueprint = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    blueprint["benchmark_info"] = context["benchmark_info"]
    blueprint["dataset"] = {"dataset_id": context["selected_dataset_id"]}
    plan = generate_task_plan(blueprint, context["dataset_profile"])
    blueprint["task_plan"] = plan
    path = _save(context, blueprint)
    apply_task_plan(path, plan, context["dataset_profile"])
    return result(
        "success",
        "Complete task plan generated",
        working_blueprint_path=path,
        plan_binding=[context["annotation_path"], context["dataset_access_manifest"]["revision"]],
    )


def revise_task(context, *, task_id: str, feedback: str):
    """Revise a single parent task."""
    current = _read(context)
    base = json.loads(get_settings().paths.blueprint_template.read_text(encoding="utf-8"))
    base.update(benchmark_info=context["benchmark_info"], dataset=current["dataset"])
    plan = revise_task_plan(
        base,
        context["dataset_profile"],
        current["task_plan"],
        task_id,
        feedback,
    )
    base["task_plan"] = plan
    path = _save(context, base)
    apply_task_plan(path, plan, context["dataset_profile"])
    updated = _read({"working_blueprint_path": path})
    parent = task_id
    for task_id, task in updated["tasks"].items():
        if task.get("parent_task_id", task_id) != parent and task_id in current["tasks"]:
            updated["tasks"][task_id] = current["tasks"][task_id]
    return result(
        "success",
        "Task revised; reallocate and design changed tasks",
        working_blueprint_path=_save(context, updated),
    )


def allocate(context, *, counts: dict[str, int] | None = None):
    """Allocate sample counts with the existing weighted algorithm."""
    blueprint = _read(context)
    counts = distribute_weighted_sample_counts(
        blueprint["tasks"], context["benchmark_info"]["sample_count"], counts
    )
    for task_id, count in counts.items():
        blueprint["tasks"][task_id]["sample_count"] = count
    return result(
        "success", "Sample counts allocated", working_blueprint_path=_save(context, blueprint)
    )


def select_materials(context, *, files: list[str]):
    """Choose exact discovered PDF paths. Available paths are in state.materials."""
    available = {m["file"] for m in discover_medical_materials(get_settings().paths.materials_root)}
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(f, str) and f in available for f in files)
    ):
        raise ValueError("Select existing PDF paths")
    updates = {"selected_material_files": list(dict.fromkeys(files))}
    if files != context.get("selected_material_files"):
        blueprint = _read(context)
        for task in blueprint["tasks"].values():
            for field in RAG_DESIGN_FIELDS:
                task.pop(field, None)
        updates["working_blueprint_path"] = _save(context, blueprint)
    return result(
        "success",
        "Medical references selected; changed references require fresh designs",
        **updates,
    )


def medical_design(context, *, task_id: str, feedback: str = ""):
    """Retrieve evidence and design one task."""
    blueprint = _read(context)
    if task_id not in blueprint["tasks"] or not context.get("selected_material_files"):
        raise ValueError("A planned task and selected materials are required")
    rag = LocalPdfRag(
        get_settings().paths.materials_root,
        get_settings().paths.rag_cache_root,
        context["selected_material_files"],
    )
    design = design_task(
        rag,
        context["benchmark_info"],
        task_id,
        blueprint["tasks"][task_id],
        dataset_profile=context["dataset_profile"],
        dataset_access_manifest=context["dataset_access_manifest"],
        feedback=feedback,
    )
    blueprint["tasks"][task_id].update(design)
    blueprint["tasks"][task_id].pop("question_templates", None)
    return result(
        "success", "Medical task design ready", working_blueprint_path=_save(context, blueprint)
    )


def finish(context):
    blueprint = _read(context)
    if (
        blueprint.get("dataset", {}).get("dataset_id") != context.get("selected_dataset_id")
        or not context.get("dataset_profile")
        or not context.get("dataset_access_manifest")
        or "task_plan" not in blueprint
    ):
        raise ValueError("Profile and plan the current dataset before finishing")
    if context.get("plan_binding") != [
        context.get("annotation_path"),
        context["dataset_access_manifest"].get("revision"),
    ]:
        raise ValueError("Task plan belongs to a different dataset revision; replan first")
    tasks = blueprint["tasks"]
    counts = [t.get("sample_count") for t in tasks.values()]
    if (
        not tasks
        or any(type(n) is not int or n < 0 for n in counts)
        or sum(counts) != context["benchmark_info"]["sample_count"]
    ):
        raise ValueError("Allocate the requested sample count before finishing")
    for task in tasks.values():
        if task.get("sample_count", 0) and not all(
            k in task for k in ("generation_logic", "medical_task_description", "citations")
        ):
            raise ValueError("Every active task needs a medical design")
    blueprint["selected_material_files"] = context["selected_material_files"]
    artifact = write_versioned_json(context["run_directory"], "blueprints", blueprint)
    return result("complete", "Task planning complete", working_blueprint_path=artifact["path"])


def describe(context):
    try:
        catalog = public_dataset_entries(load_dataset_catalog())
    except ValueError as exc:
        catalog = {"unavailable": str(exc)}
    path = context.get("working_blueprint_path")
    return {
        "requirement": context["user_requirement"],
        "benchmark_info": context["benchmark_info"],
        "dataset": context.get("selected_dataset_id"),
        "profile": context.get("dataset_profile"),
        "blueprint": _read(context) if path and Path(path).is_file() else None,
        "catalog": catalog,
        "materials": discover_medical_materials(get_settings().paths.materials_root),
        "selected_materials": context.get("selected_material_files"),
        "feedback": context.get("feedback"),
    }


def compile_task_planning_agent(*, checkpointer=None):
    return compile_agent(
        "TaskPlanningAgent",
        "Plan medical benchmark tasks. Use tools for dataset selection, profiling, complete task planning, "
        "allocation and medical design. Reuse completed work. Every nonzero task needs a medical design. "
        "Use ask_user for unresolved semantics, not routine confirmation.",
        {
            "select_dataset": select_dataset,
            "profile_dataset": profile_dataset,
            "plan_tasks": plan_tasks,
            "revise_task": revise_task,
            "allocate": allocate,
            "select_materials": select_materials,
            "medical_design": medical_design,
        },
        describe,
        finish,
        args_schemas={
            "select_dataset": SelectDatasetArguments,
            "profile_dataset": ProfileArguments,
            "revise_task": ReviseTaskArguments,
            "allocate": AllocateArguments,
            "select_materials": MaterialsArguments,
            "medical_design": MedicalDesignArguments,
        },
        checkpointer=checkpointer,
    )
