"""Initialization of a Coding Session preparation run.

This module owns run-frozen Source, backend and bundle context. The preparation
graph consumes this immutable context.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from medbenchagent.data.contracts import Source
from medbenchagent.data.dataset_catalog import validate_dataset_id
from medbenchagent.data.execution import data_dependency_identity, docker_image_id, select_backend
from medbenchagent.data.sources import digest_file, inventory_source
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


def _coding_worker_identity() -> dict[str, str]:
    settings = get_settings().data_processing
    venv = Path(settings.get("coding_worker_venv", ".venv-coding")).resolve()
    executable = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = Path(__file__).resolve().parents[3] / "config" / "coding-worker.requirements.txt"
    identity = {
        "venv": str(venv),
        "python": str(executable),
        "requirements_digest": digest_file(requirements) if requirements.is_file() else "",
    }
    if executable.is_file():
        try:
            probe = subprocess.run(
                [str(executable), "-c", "import importlib.metadata,sys; print(sys.version); print(importlib.metadata.version('openhands-sdk'))"],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            probe = None
        if probe is not None and probe.returncode == 0:
            lines = probe.stdout.splitlines()
            identity["python_version"] = lines[0] if lines else ""
            identity["openhands_version"] = lines[1] if len(lines) > 1 else ""
    return identity


def initialize_preparation(context: dict, sources, dataset_id: str) -> dict:
    sources = [Source.model_validate(source) for source in sources]
    validate_dataset_id(dataset_id)
    if not sources or len({source.id for source in sources}) != len(sources):
        raise ValueError("Provide nonempty sources with unique IDs within this dataset")
    runtime = dict(context.get("execution_runtime") or select_backend(get_settings().data_processing))
    if runtime.get("backend") == "local":
        runtime["coding_worker"] = _coding_worker_identity()
    marker = Path(context["operation_directory"]) / "preparation.json"
    def configured_bundle() -> Path:
        return get_settings().paths.prepared_datasets_root / dataset_id

    def overlaps_source(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved.is_relative_to(Path(source.root).resolve()) for source in sources)

    if marker.is_file():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["sources"] != [source.model_dump() for source in sources] or saved["dataset_id"] != dataset_id:
            raise ValueError("Preparation inputs changed; use a new operation")
        bundle = Path(saved["bundle"])
        if saved["runtime"] != runtime:
            raise ValueError("Frozen execution runtime changed; start a new run")
        # A failed older attempt may have persisted an output path beneath the
        # source before the guard ran. Rebind that marker to the configured
        # prepared root so clarification/retry cannot repeat the stale path.
        if overlaps_source(bundle):
            bundle = configured_bundle()
            if overlaps_source(bundle):
                raise ValueError(
                    "paths.prepared_datasets_root must be outside every configured source root; "
                    f"configured output {bundle} overlaps source data"
                )
            atomic_write_json(marker, {"sources": [source.model_dump() for source in sources],
                                       "dataset_id": dataset_id, "bundle": str(bundle), "runtime": runtime})
    else:
        bundle = configured_bundle()
        if overlaps_source(bundle):
            raise ValueError(
                "paths.prepared_datasets_root must be outside every configured source root; "
                f"configured output {bundle} overlaps source data"
            )
        atomic_write_json(marker, {"sources": [source.model_dump() for source in sources],
                                   "dataset_id": dataset_id, "bundle": str(bundle), "runtime": runtime})
    run_workspace = Path(context.get("run_directory") or Path(context["operation_directory"]).parent) / "preparation" / dataset_id
    run_workspace.mkdir(parents=True, exist_ok=True)
    runtime_path = run_workspace / "inputs" / "execution_runtime.json"
    if runtime_path.is_file():
        data_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if any(data_runtime.get(key) != value for key, value in runtime.items()):
            raise ValueError("Prepared data runtime differs from the frozen run")
    else:
        component = ({"image_id": docker_image_id(get_settings().data_processing["image"])}
                      if runtime["backend"] == "docker" else {"packages": data_dependency_identity()})
        data_runtime = {**runtime, **component}
        atomic_write_json(runtime_path, data_runtime)
    manifest_root = run_workspace if context.get("run_directory") else bundle
    manifest_base = manifest_root / ("inputs/source_manifests" if context.get("run_directory") else "source_manifests")
    for source in sources:
        if bundle.resolve().is_relative_to(Path(source.root).resolve()):
            raise ValueError("Configured prepared output overlaps a source directory")
        manifest = manifest_base / source.id
        if not all((manifest / name).is_file() for name in ("files.jsonl", "source.json", "used.sqlite")):
            inventory_source(source, manifest)
    prepared = dict(context, sources=[source.model_dump() for source in sources], dataset_id=dataset_id,
                    dataset_scope=dataset_id, agent_role="DataProcessingAgent", execution_runtime=runtime,
                    data_runtime=data_runtime,
                    coding_worker_max_iterations=get_settings().data_processing["coding_worker_max_iterations"],
                    coding_worker_timeout_seconds=get_settings().data_processing["coding_worker_timeout_seconds"],
                    preparation_attempt_limits=get_settings().data_processing.get(
                        "preparation_attempt_limits", {"understand": 2, "prepare": 2, "mapping_revision_cycles": 1}
                    ),
                    preparation_stage_budgets=get_settings().data_processing.get("preparation_stage_budgets", {}),
                    bundle_root=str(bundle), source_scope=None,
                     source_roots={source.id: source.root for source in sources},
                     source_manifests={source.id: str(manifest_base / source.id) for source in sources},
                     preparation_root=str(run_workspace))
    publish(prepared, "SourceSelection", {"sources": prepared["sources"]}, ())
    return prepared
