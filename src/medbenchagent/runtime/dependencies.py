"""Run-local immutable artifacts and their direct version dependencies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.runtime.construction import record_event

CONTRACT_VERSION = "agent-runtime"

DATASET_KINDS = frozenset(
    {
        "Dataset",
        "DatasetProfile",
        "DatasetCard",
        "DatasetCandidates",
        "DatasetSelection",
        "PlanningDraft",
        "Plan",
        "PlanningDelegation",
        "Materials",
        "Templates",
        "Generation",
        "GenerationCheck",
        "SourceSelection",
        "Documentation",
        "Allocation",
        "Delivery",
        "PreparedCandidates",
    }
)


def artifact_key(context, kind):
    if kind.startswith("dataset/"):
        return kind
    dataset = context.get("dataset_scope")
    return (
        f"dataset/{dataset}/{kind}" if dataset and kind in DATASET_KINDS else kind
    )


def reference_digests(references):
    from medbenchagent.data.sources import digest_file

    digests = {}

    # Prepared access manifests already bind the full annotation contents. Reuse that
    # digest instead of scanning a large annotation again for every publication.
    def known(value):
        if isinstance(value, dict):
            if "cases_path" in value and "annotation_digest" in value:
                digests[str(Path(value["cases_path"]).resolve())] = value["annotation_digest"]
            for child in value.values():
                known(child)

    known(references)

    def visit(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, name)
        elif isinstance(value, str) and (
            key.endswith("_path")
            or key in {"path", "blueprint", "raw", "audited", "report", "cases"}
        ):
            path = Path(value)
            if path.is_file() and path.suffix in {".json", ".jsonl", ".py", ".toml"}:
                canonical = str(path.resolve())
                if canonical not in digests:
                    digests[canonical] = digest_file(path)

    visit(references)
    return digests


DEPENDENCIES = {
    "Dataset": ("DataConfig",),
    "Plan": ("Requirement", "Dataset", "Materials", "PlanningConfig"),
    "Templates": ("Plan", "TemplateConfig"),
    # Task Plan now owns the reviewed question templates and resolver
    # contract.  Annotation generation therefore has no separate Templates
    # artifact dependency.
    "Generation": ("Dataset", "Plan", "GenerationConfig", "Allocation"),
}


def index_path(context):
    return Path(context["run_directory"]) / "control" / "artifacts.json"


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
    identity = {
        "kind": kind,
        "references": references,
        "dependencies": deps,
        "content_digests": reference_digests(references),
    }
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
    path = Path(context["run_directory"]) / "control" / "contract.json"
    if not path.is_file():
        raise ValueError("This checkpoint uses an obsolete runtime; start a new run")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("contract") != CONTRACT_VERSION:
        raise ValueError("Code or runtime contract changed; start a new run")
    current_code = code_identity()
    saved_code = saved.get("code_at_start", saved.get("code", {}))
    code_changed = saved_code != current_code
    event_path = context.get("construction_events_path")
    if event_path:
        record_event(
            event_path,
            "resume_compatibility",
            run_id=context.get("run_id"),
            code_changed=code_changed,
            start_commit=saved_code.get("commit") if isinstance(saved_code, dict) else None,
            current_commit=current_code.get("commit"),
            source_diff=bool(current_code.get("source_diff")),
            contract=CONTRACT_VERSION,
        )
    frozen = json.loads(Path(context["effective_config_path"]).read_text(encoding="utf-8"))
    if frozen.get("execution_runtime") != context.get("execution_runtime"):
        raise ValueError("Frozen execution runtime changed; start a new run")
    from medbenchagent.runtime.run_artifacts import verify_catalog_snapshot

    catalog_snapshot = context.get("task_catalog_snapshot_path")
    catalog_digest = context.get("task_catalog_snapshot_digest")
    if not catalog_snapshot or not catalog_digest:
        raise ValueError("Frozen task catalog is unavailable; this run cannot resume")
    verify_catalog_snapshot(catalog_snapshot, catalog_digest)
    from medbenchagent.data.sources import verify_source

    for source in Path(context["run_directory"]).rglob("source_manifests/*/source.json"):
        verify_source(source.parent)
    from medbenchagent.data.dataset_catalog import load_prepared
    from medbenchagent.settings import run_settings

    entries = dict(context.get("prepared_entries", {}))
    entries.update(
        {key: asset["entry"] for key, asset in context.get("dataset_assets", {}).items()}
    )
    if entries:
        with run_settings(context["effective_config_path"]):
            for entry in entries.values():
                load_prepared(entry)


def initialize_dependencies(context, settings):
    snapshot = settings.snapshot()["settings"]
    configs = {
        "DataConfig": {
            "profile": snapshot["dataset_profile"],
            "preparation": snapshot["data_processing"],
            "runtime": context["execution_runtime"],
        },
        "PlanningConfig": {
            "rag": snapshot["rag"],
            "model": snapshot["llm"]["model"],
            "temperature": snapshot["llm"]["default_temperature"],
        },
        "AllocationConfig": {"policy": "concept-dataset-leaf", "model": settings.llm.model},
        "TemplateConfig": {
            "count": settings.generation.question_template_count,
            "temperature": context["question_temperature"],
            "model": settings.llm.model,
        },
        "GenerationConfig": {
            key: context.get(key)
            for key in ("seed", "max_tasks_per_image", "max_total_tasks_per_image")
        },
        "FinalizationConfig": {"seed": context["seed"], "policy": "final-task"},
    }
    for kind, value in configs.items():
        path = Path(context["run_directory"]) / "control" / "config" / f"{kind}.json"
        atomic_write_json(path, value)
        publish(context, kind, {"path": str(path)}, ())
    atomic_write_json(
        Path(context["run_directory"]) / "control" / "execution_runtime.json",
        context["execution_runtime"],
    )
    atomic_write_json(
        Path(context["run_directory"]) / "control" / "contract.json",
        {"contract": CONTRACT_VERSION, "code_at_start": code_identity()},
    )

