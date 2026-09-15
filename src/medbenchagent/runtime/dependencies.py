"""Run-local immutable artifacts and their direct version dependencies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from medbenchagent.runtime.run_artifacts import atomic_write_json

CONTRACT_VERSION = "agent-runtime.v5"

SOURCE_KINDS = frozenset({"Source", "Inventory", "Mapping", "AnnotationScript", "Annotation",
                          "ImageInventory", "ImagePlan", "NormalizedSource"})


def artifact_key(context, kind):
    scope = context.get("source_scope")
    return f"source/{scope}/{kind}" if scope and kind in SOURCE_KINDS else kind


DEPENDENCIES = {
    "Dataset": ("DataConfig",),
    "Plan": ("Requirement", "Dataset", "Materials", "PlanningConfig"),
    "Templates": ("Plan", "TemplateConfig"),
    "Generation": ("Dataset", "Plan", "Templates", "GenerationConfig"),
    "Audit": ("Generation", "Dataset", "Plan", "AuditConfig"),
}


def index_path(context):
    return Path(context["run_directory"]) / ".runtime" / "artifacts.json"


def read_index(context):
    path = index_path(context)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"versions": {}}


def require(context, *kinds):
    """Reject absent or stale artifacts, including transitive upstream revisions."""
    index = read_index(context)["versions"]
    current = context.get("artifact_versions", {})

    def check(kind):
        identifier = current.get(kind)
        if identifier not in index:
            raise ValueError(f"{kind} is required; complete its producing stage first")
        record = index[identifier]
        for dependency, version in record["dependencies"].items():
            if current.get(dependency) != version:
                raise ValueError(f"{kind} is stale after {dependency} changed; rebuild {kind}")
            check(dependency)
        return record

    return {kind: check(artifact_key(context, kind)) for kind in kinds}


def publish(context, kind, references, dependencies=None):
    """Append a version; replay of the same operation and inputs returns that version."""
    kinds = DEPENDENCIES.get(kind, ()) if dependencies is None else dependencies
    kinds = tuple(artifact_key(context, k) for k in kinds)
    kind = artifact_key(context, kind)
    require(context, *kinds)
    current = dict(context.get("artifact_versions", {}))
    deps = {key: current[key] for key in kinds}
    index = read_index(context)
    identity = {"kind": kind, "references": references, "dependencies": deps}
    existing = next(
        (
            key
            for key, value in index["versions"].items()
            if {k: value[k] for k in identity} == identity
        ),
        None,
    )
    identifier = existing or f"{kind}:{len(index['versions']) + 1}"
    index["versions"][identifier] = dict(identity, id=identifier)
    current[kind] = identifier
    index["current"] = current
    atomic_write_json(index_path(context), index)
    context["artifact_versions"] = current
    return {"artifact_versions": current}


def code_identity():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "source_diff": git("diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"),
        "untracked_sources": {
            name: (root / name).read_text(encoding="utf-8")
            for name in git("ls-files", "--others", "--exclude-standard", "src").splitlines()
        },
    }


def verify_resume(context):
    path = Path(context["run_directory"]) / ".runtime" / "contract.json"
    if not path.is_file():
        raise ValueError("This checkpoint uses an obsolete runtime; start a new run")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != {"contract": CONTRACT_VERSION, "code": code_identity()}:
        raise ValueError("Code or runtime contract changed; start a new run")
    from medbenchagent.data.sources import verify_source
    for source in Path(context["run_directory"]).rglob("source_manifests/*/source.json"):
        verify_source(source.parent)


def initialize_dependencies(context, settings):
    snapshot = settings.snapshot()["settings"]
    configs = {
        "DataConfig": {
            "profile": snapshot["dataset_profile"],
            "preparation": snapshot["data_processing"],
        },
        "PlanningConfig": {
            "rag": snapshot["rag"],
            "model": snapshot["llm"]["model"],
            "temperature": snapshot["llm"]["default_temperature"],
        },
        "TemplateConfig": {
            "count": settings.generation.question_template_count,
            "temperature": context["question_temperature"],
            "model": settings.llm.model,
        },
        "GenerationConfig": {key: context[key] for key in ("seed", "max_tasks_per_image")},
        "AuditConfig": {"seed": context["seed"], "model": settings.llm.model},
    }
    for kind, value in configs.items():
        path = Path(context["run_directory"]) / ".runtime" / "config" / f"{kind}.json"
        atomic_write_json(path, value)
        publish(context, kind, {"path": str(path)}, ())
    atomic_write_json(
        Path(context["run_directory"]) / ".runtime" / "contract.json",
        {"contract": CONTRACT_VERSION, "code": code_identity()},
    )


def snapshot_dataset(annotation, destination, *, dataset_id="imported"):
    """Import standard PNG/JPEG inputs through the same quality and publication engine.

    This fixed local copy path executes no generated code. Raw conversion stays in Docker.
    """
    from medbenchagent.data.bundle import normalize_source, publish_bundle
    from medbenchagent.data.contracts import Source, HarmonizationPlan
    from medbenchagent.data.sources import (
        inventory_source, track_files, verify_source, compare_outputs, remove_work_directory,
    )
    from medbenchagent.data.image_assets import iter_image_manifest

    annotation, destination = Path(annotation).resolve(), Path(destination).resolve()
    source = Source(id=dataset_id, kind="annotation", root=str(annotation.parent), annotation=annotation.name)
    source_manifest = destination / "source_manifests" / dataset_id
    if (destination / "ready.json").exists():
        verify_source(source_manifest)
        return destination / "annotation.jsonl"
    if destination.is_relative_to(annotation.parent):
        raise ValueError("Dataset output must not be beneath the source root")
    inventory_source(source, source_manifest)
    from medbenchagent.data.image_conversion import detect_format
    for entry in iter_image_manifest(annotation.parent / "image_manifest.jsonl"):
        if not entry.png_path.is_file():
            continue  # Missing files are reported by normalization as local quarantine.
        detected = detect_format(entry.png_path)
        if detected is not None and detected not in {"raster:png", "raster:jpeg"}:
            raise ValueError("Standard imports require PNG/JPEG; use prepare_data for raw conversion")
    track_files(source_manifest, [annotation.name, "image_manifest.jsonl"])
    track_files(source_manifest, (e.relative_png_path for e in iter_image_manifest(annotation.parent / "image_manifest.jsonl")
                                  if e.png_path.is_file()))
    folder = destination / "sources" / dataset_id
    normalize_source(source.root, annotation, source, folder)
    repeat = destination / "repeat"
    normalize_source(source.root, annotation, source, repeat)
    compare_outputs(folder, repeat)
    remove_work_directory(repeat, destination)
    verify_source(source_manifest)
    publish_bundle(destination, {dataset_id: folder}, HarmonizationPlan(),
                   verify_inputs=lambda: verify_source(source_manifest))
    verify_source(source_manifest)
    return destination / "annotation.jsonl"
