"""Task planning with bounded evidence queries, family drafts and global reconciliation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from medbenchagent.data.dataset import prepare_annotation_dataset
from medbenchagent.data.dataset_catalog import (
    dataset_by_id,
    load_dataset_catalog,
    public_dataset_entries,
    recommend_dataset,
    resolve_annotation_path,
)
from medbenchagent.runtime.dependencies import publish, require, snapshot_dataset
from medbenchagent.runtime.harness import compile_agent, result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.tool_contracts import (
    AllocateArguments,
    MaterialsArguments,
    MedicalDesignArguments,
    ProfileArguments,
    SelectDatasetArguments,
    PageArguments,
    FieldArguments,
    TaskArguments,
    FamilyArguments,
    MaterialArguments,
    SearchArguments,
)
from medbenchagent.rag.pdf import LocalPdfRag, design_task, discover_medical_materials
from medbenchagent.planning.sample_counts import distribute_weighted_sample_counts
from medbenchagent.planning.task_screening import (
    apply_task_plan,
    generate_task_plan,
    coordinate_task_plan,
)
from medbenchagent.settings import get_settings


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _base(context):
    blueprint = _read(get_settings().paths.blueprint_template)
    blueprint.update(
        benchmark_info=context["benchmark_info"],
        dataset={"dataset_id": context.get("selected_dataset_id")},
    )
    return blueprint


def _profile(context):
    require(context, "Dataset")
    return _read(context["dataset_access_manifest"]["profile_path"])


def _save(context, blueprint):
    blueprint["materials_version"] = context["artifact_versions"]["Materials"]
    path = Path(context["operation_directory"]) / "blueprint.json"
    atomic_write_json(path, blueprint)
    publish(
        context,
        "BlueprintDraft",
        {"path": str(path)},
        ("Dataset", "Requirement", "Materials", "PlanningDraft", "PlanningConfig"),
    )
    return str(path)


def select_dataset(context, *, dataset_id=None, annotation_path=None):
    """Select standard inputs; omit both arguments to recommend catalog data."""
    require(context, "Requirement")
    if (
        dataset_id is not None
        and dataset_id == context.get("selected_dataset_id")
        and (
            annotation_path is None
            or annotation_path == context.get("dataset_entry", {}).get("annotation_path")
        )
    ):
        require(context, "DatasetSelection")
        return result("success", "This dataset is already selected in the run")
    if annotation_path:
        if dataset_id is None:
            raise ValueError("Direct input requires dataset_id")
        entry = {
            "id": dataset_id,
            "name": dataset_id,
            "annotation_path": annotation_path,
            "description": "Standard local dataset",
            "available": True,
        }
    else:
        catalog = load_dataset_catalog()
        dataset_id = (
            dataset_id or recommend_dataset(context["benchmark_info"], catalog)["dataset_id"]
        )
        entry = dataset_by_id(catalog, dataset_id)
    path = snapshot_dataset(
        resolve_annotation_path(entry), Path(context["operation_directory"]) / "input"
    )
    versions = publish(
        context, "DatasetSelection", {"path": str(path), "dataset_id": dataset_id}, ()
    )
    return result(
        "success",
        "Dataset input snapshot selected; build its profile next",
        selected_dataset_id=dataset_id,
        annotation_path=str(path),
        dataset_entry=entry,
        dataset_access_manifest=None,
        **versions,
    )


def profile_dataset(context, *, revision="1", refresh=False):
    """Build this run's dataset cache; refresh creates a new immutable Dataset version."""
    require(context, "DatasetSelection")
    if not refresh and context.get("dataset_access_manifest"):
        require(context, "Dataset")
        return result("success", "Current Dataset is already prepared")
    profile, access = prepare_annotation_dataset(
        context["annotation_path"],
        Path(context["operation_directory"]) / "cache",
        revision=revision,
    )
    from medbenchagent.data.validation import load_annotation_jsonl, validate_dataset
    from medbenchagent.data.image_assets import load_image_manifest

    validate_dataset(
        load_annotation_jsonl(context["annotation_path"]),
        load_image_manifest(Path(context["annotation_path"]).parent / "image_manifest.jsonl"),
    )
    versions = publish(
        context,
        "Dataset",
        {"annotation": context["annotation_path"], "access": access},
        ("DataConfig", "DatasetSelection"),
    )
    return result(
        "success",
        f"Dataset ready: {profile['total_images']} images, {len(profile['fields'])} fields",
        dataset_access_manifest=access,
        **versions,
    )


def list_tasks(context, *, offset=0, limit=20, query=""):
    """Page through predefined task IDs, families and short descriptions."""
    tasks = [
        {"task_id": key, "category": value["category"], "description": value["description"]}
        for key, value in _base(context)["tasks"].items()
        if query.casefold() in (key + " " + value["category"]).casefold()
    ]
    return result(
        "success",
        json.dumps(
            {"total": len(tasks), "tasks": tasks[offset : offset + limit]}, ensure_ascii=False
        ),
    )


def read_task(context, *, task_id):
    """Read one task definition, family draft and current design."""
    tasks = _base(context)["tasks"]
    if task_id not in tasks:
        raise ValueError("Unknown predefined task")
    draft = _read(context["draft_path"]) if context.get("draft_path") else {}
    blueprint = (
        _read(context["working_blueprint_path"])
        if context.get("working_blueprint_path")
        and Path(context["working_blueprint_path"]).is_file()
        else {}
    )
    return result(
        "success",
        json.dumps(
            {
                "definition": tasks[task_id],
                "draft": draft.get("tasks", {}).get(task_id),
                "design": blueprint.get("tasks", {}).get(task_id),
            },
            ensure_ascii=False,
        ),
    )


def list_fields(context, *, offset=0, limit=20, query=""):
    """Page field paths and availability; text search helps discovery but never drops tasks."""
    fields = _profile(context)["fields"]
    rows = [
        {"path": key, "types": value["types"], "non_empty_images": value["non_empty_images"]}
        for key, value in fields.items()
        if query.casefold() in key.casefold()
    ]
    return result(
        "success", json.dumps({"total": len(rows), "fields": rows[offset : offset + limit]})
    )


def read_fields(context, *, fields):
    """Read selected field distributions and their object/array structure."""
    return result("success", json.dumps(_subset(_profile(context), fields), ensure_ascii=False))


def _subset(profile, fields):
    unknown = set(fields) - set(profile["fields"])
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    return {
        "total_images": profile["total_images"],
        "fields": {key: profile["fields"][key] for key in fields},
        "structures": [
            item
            for item in profile["structures"]
            if any(field.startswith(item["path"]) for field in fields)
        ],
    }


def screen_family(context, *, category, fields, feedback=""):
    """Screen or revise one task family using queried evidence; save an unpublishable draft."""
    require(context, "Dataset", "Requirement", "Materials")
    base = _base(context)
    tasks = {key: value for key, value in base["tasks"].items() if value["category"] == category}
    if not tasks:
        raise ValueError("Unknown task category")
    previous = {}
    if context.get("draft_path"):
        try:
            require(context, "PlanningDraft")
            previous = _read(context["draft_path"])["tasks"]
        except ValueError:
            previous = {}
    plan = generate_task_plan(
        {"tasks": tasks},
        _subset(_profile(context), fields),
        allow_empty=True,
        feedback=feedback + "\n" + json.dumps(context.get("clarification", {})),
    )
    merged = {**previous, **plan["tasks"]}
    path = Path(context["operation_directory"]) / "family_drafts.json"
    atomic_write_json(path, {"tasks": merged})
    versions = publish(
        context,
        "PlanningDraft",
        {"path": str(path)},
        ("Dataset", "Requirement", "Materials", "PlanningConfig"),
    )
    return result(
        "success",
        f"Family draft saved; {len(merged)}/{len(base['tasks'])} tasks decided",
        draft_path=str(path),
        **versions,
    )


def coordinate(context):
    """Globally reconcile all family decisions and expand the resulting complete Task Plan."""
    require(context, "PlanningDraft")
    base = _base(context)
    profile = _profile(context)
    plan = coordinate_task_plan(base, profile, _read(context["draft_path"])["tasks"])
    old = (
        _read(context["working_blueprint_path"])
        if context.get("working_blueprint_path")
        and Path(context["working_blueprint_path"]).is_file()
        else {}
    )
    path = _save(context, base)
    apply_task_plan(path, plan, profile)
    updated = _read(path)
    for key, task in updated["tasks"].items():
        parent = task.get("parent_task_id", key)
        if (
            old.get("materials_version") == context["artifact_versions"]["Materials"]
            and old.get("task_plan", {}).get("tasks", {}).get(parent) == plan["tasks"][parent]
            and key in old.get("tasks", {})
        ):
            updated["tasks"][key] = old["tasks"][key]
    atomic_write_json(path, updated)
    return result(
        "success",
        "Global plan coordinated; allocate and design active tasks",
        working_blueprint_path=path,
        artifact_versions=context["artifact_versions"],
    )


def allocate(context, *, counts=None):
    """Allocate requested samples using the existing weighted algorithm."""
    require(context, "BlueprintDraft")
    blueprint = _read(context["working_blueprint_path"])
    counts = distribute_weighted_sample_counts(
        blueprint["tasks"], context["benchmark_info"]["sample_count"], counts
    )
    for key, count in counts.items():
        blueprint["tasks"][key]["sample_count"] = count
    path = _save(context, blueprint)
    return result(
        "success",
        "Sample counts allocated",
        working_blueprint_path=path,
        artifact_versions=context["artifact_versions"],
    )


def list_materials(context, *, offset=0, limit=20, query=""):
    """Page available run-local medical references."""
    files = [
        m
        for m in discover_medical_materials(get_settings().paths.materials_root)
        if query.casefold() in m["file"].casefold()
    ]
    return result(
        "success", json.dumps({"total": len(files), "materials": files[offset : offset + limit]})
    )


def read_material(context, *, file):
    """Read one PDF's metadata and page count without dumping its contents."""
    from pypdf import PdfReader

    available = {m["file"] for m in discover_medical_materials(get_settings().paths.materials_root)}
    if file not in available:
        raise ValueError("Unknown PDF")
    reader = PdfReader(get_settings().paths.materials_root / file)
    return result(
        "success",
        json.dumps(
            {"file": file, "pages": len(reader.pages), "metadata": str(reader.metadata)[:2000]},
            ensure_ascii=False,
        ),
    )


def select_materials(context, *, files):
    """Select medical references and retain an immutable copy for this run."""
    available = {m["file"] for m in discover_medical_materials(get_settings().paths.materials_root)}
    if not set(files) <= available:
        raise ValueError("Select existing PDF paths")
    files = list(dict.fromkeys(files))
    if files == context.get("selected_material_files"):
        require(context, "Materials")
        return result("success", "These references are already selected")
    root = Path(context["operation_directory"]) / "materials"
    for file in files:
        destination = root / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(get_settings().paths.materials_root / file, destination)
    versions = publish(context, "Materials", {"root": str(root), "files": files}, ())
    return result(
        "success",
        "References selected; reconcile affected planning evidence",
        selected_material_files=files,
        materials_root=str(root),
        **versions,
    )


def _rag(context):
    require(context, "Materials")
    return LocalPdfRag(
        context["materials_root"],
        Path(context["run_directory"]) / ".runtime/rag",
        context["selected_material_files"],
    )


def search_materials(context, *, query, limit=5):
    """Retrieve relevant paragraphs from selected references."""
    return result("success", json.dumps(_rag(context).retrieve(query, limit), ensure_ascii=False))


def medical_design(context, *, task_id, feedback=""):
    """Design one expanded task from its supporting fields and selected medical references."""
    require(context, "BlueprintDraft")
    blueprint = _read(context["working_blueprint_path"])
    if task_id not in blueprint["tasks"]:
        raise ValueError("Unknown expanded task")
    task = blueprint["tasks"][task_id]
    design = design_task(
        _rag(context),
        context["benchmark_info"],
        task_id,
        task,
        dataset_profile=_subset(_profile(context), task["supporting_fields"]),
        dataset_access_manifest=context["dataset_access_manifest"],
        feedback=feedback,
    )
    task.update(design)
    task.pop("question_templates", None)
    path = _save(context, blueprint)
    return result(
        "success",
        "Medical design ready",
        working_blueprint_path=path,
        artifact_versions=context["artifact_versions"],
    )


def finish(context):
    """Publish a fully coordinated, allocated and medically designed Plan version."""
    require(context, "BlueprintDraft")
    blueprint = _read(context["working_blueprint_path"])
    tasks = blueprint["tasks"]
    counts = [task.get("sample_count") for task in tasks.values()]
    if (
        not tasks
        or any(type(n) is not int or n < 0 for n in counts)
        or sum(counts) != context["benchmark_info"]["sample_count"]
    ):
        raise ValueError("Allocate the requested sample count before finishing")
    for task in tasks.values():
        if task["sample_count"] and not all(
            key in task for key in ("generation_logic", "medical_task_description", "citations")
        ):
            raise ValueError("Every active task needs a medical design")
    versions = publish(
        context,
        "Plan",
        {"blueprint": context["working_blueprint_path"]},
        ("Requirement", "Dataset", "Materials", "PlanningConfig", "BlueprintDraft"),
    )
    return result("complete", "Plan published", **versions)


def describe(context):
    drafts = _read(context["draft_path"])["tasks"] if context.get("draft_path") else {}
    blueprint = (
        _read(context["working_blueprint_path"])
        if context.get("working_blueprint_path")
        and Path(context["working_blueprint_path"]).is_file()
        else {}
    )
    return {
        "requirement": context["user_requirement"][:2000],
        "benchmark_info": context["benchmark_info"],
        "dataset": context.get("selected_dataset_id"),
        "versions": context.get("artifact_versions", {}),
        "families": sorted({task["category"] for task in _base(context)["tasks"].values()}),
        "draft_count": len(drafts),
        "draft_reference": context.get("draft_path"),
        "active_tasks": [
            {"id": key, "count": task.get("sample_count"), "designed": "generation_logic" in task}
            for key, task in blueprint.get("tasks", {}).items()
        ],
        "feedback": context.get("feedback"),
        "clarification": context.get("clarification"),
        "dataset_catalog": public_dataset_entries(load_dataset_catalog())
        if not context.get("selected_dataset_id")
        else None,
    }


def compile_task_planning_agent(*, checkpointer=None):
    tools = {
        function.__name__: function
        for function in (
            select_dataset,
            profile_dataset,
            list_tasks,
            read_task,
            list_fields,
            read_fields,
            screen_family,
            coordinate,
            allocate,
            list_materials,
            read_material,
            select_materials,
            search_materials,
            medical_design,
        )
    }
    return compile_agent(
        "TaskPlanningAgent",
        "Plan predefined medical benchmark tasks. Select/profile data and references, list task families, "
        "query candidate fields and structures, then screen each family. Never drop a task solely because "
        "a keyword search failed; query more evidence or ask_user. Coordinate all family drafts globally, "
        "allocate, then design every nonzero expanded task. Revise only affected families and coordinate again. "
        "Use local ask_user for source semantics; scope global only for cross-role goals or resource decisions.",
        tools,
        describe,
        finish,
        args_schemas={
            "select_dataset": SelectDatasetArguments,
            "profile_dataset": ProfileArguments,
            "list_tasks": PageArguments,
            "read_task": TaskArguments,
            "list_fields": PageArguments,
            "read_fields": FieldArguments,
            "screen_family": FamilyArguments,
            "allocate": AllocateArguments,
            "list_materials": PageArguments,
            "read_material": MaterialArguments,
            "select_materials": MaterialsArguments,
            "search_materials": SearchArguments,
            "medical_design": MedicalDesignArguments,
        },
        checkpointer=checkpointer,
    )
