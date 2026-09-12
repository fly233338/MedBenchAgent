"""Run-local immutable artifacts and their direct version dependencies."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from medbenchagent.runtime.run_artifacts import atomic_write_json

CONTRACT_VERSION = "agent-runtime.v3"
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

    return {kind: check(kind) for kind in kinds}


def publish(context, kind, references, dependencies=None):
    """Append a version; replay of the same operation and inputs returns that version."""
    kinds = DEPENDENCIES.get(kind, ()) if dependencies is None else dependencies
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


def initialize_dependencies(context, settings):
    snapshot = settings.snapshot()["settings"]
    configs = {
        "DataConfig": {
            "profile": snapshot["dataset_profile"],
            "preparation_image": snapshot["data_processing"]["image"],
        },
        "PlanningConfig": {
            "rag": snapshot["rag"],
            "allocation": snapshot["sample_allocation"],
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


def snapshot_dataset(annotation, destination):
    """Copy exactly the selected standard dataset into this input version."""
    from medbenchagent.data.image_assets import load_image_manifest

    annotation, destination = Path(annotation), Path(destination)
    marker = destination / "ready.json"
    if marker.is_file():
        return destination / "annotation.jsonl"
    images = load_image_manifest(annotation.parent / "image_manifest.jsonl")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(annotation, destination / "annotation.jsonl")
    for entry in images.values():
        target = destination / entry.relative_png_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry.png_path, target)
    shutil.copyfile(
        annotation.parent / "image_manifest.jsonl", destination / "image_manifest.jsonl"
    )
    atomic_write_json(marker, {"source": str(annotation), "images": len(images)})
    return destination / "annotation.jsonl"
