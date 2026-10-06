"""Initialization of a Coding Session preparation run.

This module owns run-frozen Source, backend and bundle context. The preparation
graph consumes this immutable context.
"""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.data.contracts import Source
from medbenchagent.data.dataset_catalog import validate_dataset_id
from medbenchagent.data.execution import data_dependency_identity, docker_image_id, select_backend
from medbenchagent.data.sources import inventory_source
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


def initialize_preparation(context: dict, sources, dataset_id: str) -> dict:
    sources = [Source.model_validate(source) for source in sources]
    validate_dataset_id(dataset_id)
    if not sources or len({source.id for source in sources}) != len(sources):
        raise ValueError("Provide nonempty sources with unique IDs within this dataset")
    runtime = context.get("execution_runtime") or select_backend(get_settings().data_processing)
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
    for source in sources:
        if bundle.resolve().is_relative_to(Path(source.root).resolve()):
            raise ValueError("Configured prepared output overlaps a source directory")
        inventory_source(source, manifest_root / ("inputs/source_manifests" if context.get("run_directory") else "source_manifests") / source.id)
    prepared = dict(context, sources=[source.model_dump() for source in sources], dataset_id=dataset_id,
                    dataset_scope=dataset_id, agent_role="DataProcessingAgent", execution_runtime=runtime,
                    data_runtime=data_runtime,
                    coding_worker_max_iterations=get_settings().data_processing["coding_worker_max_iterations"],
                    coding_worker_timeout_seconds=get_settings().data_processing["coding_worker_timeout_seconds"],
                    preparation_no_progress_limit=get_settings().data_processing.get("preparation_no_progress_limit", 8),
                    preparation_tool_error_limit=get_settings().data_processing.get("preparation_tool_error_limit", 3),
                    preparation_transport_retries=get_settings().data_processing.get("preparation_transport_retries", 2),
                    preparation_repair_attempts=get_settings().data_processing.get("preparation_repair_attempts", 2),
                    preparation_semantic_revision_cycles=get_settings().data_processing.get("preparation_semantic_revision_cycles", 2),
                    preparation_total_seconds=get_settings().data_processing.get("preparation_total_seconds", 14_400),
                    preparation_total_iterations=get_settings().data_processing.get("preparation_total_iterations", 300),
                    preparation_stage_budgets=get_settings().data_processing.get("preparation_stage_budgets", {}),
                    bundle_root=str(bundle), source_scope=None,
                     source_roots={source.id: source.root for source in sources},
                     source_manifests={source.id: str(manifest_root / ("inputs/source_manifests" if context.get("run_directory") else "source_manifests") / source.id) for source in sources},
                     preparation_root=str(run_workspace))
    publish(prepared, "SourceSelection", {"sources": prepared["sources"]}, ())
    return prepared
