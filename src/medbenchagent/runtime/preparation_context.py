"""Initialization of a Coding Session preparation run.

This module owns run-frozen Source, backend and bundle context. It deliberately
does not expose a stage scheduler.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from medbenchagent.data.contracts import Source
from medbenchagent.data.dataset_catalog import validate_dataset_id
from medbenchagent.data.execution import data_dependency_identity, docker_image_id, select_backend
from medbenchagent.data.sources import inventory_source
from medbenchagent.runtime.dependencies import publish
from medbenchagent.runtime.run_artifacts import atomic_write_json, version_directory_name
from medbenchagent.settings import get_settings


def initialize_preparation(context: dict, sources, dataset_id: str) -> dict:
    sources = [Source.model_validate(source) for source in sources]
    validate_dataset_id(dataset_id)
    if not sources or len({source.id for source in sources}) != len(sources):
        raise ValueError("Provide nonempty sources with unique IDs within this dataset")
    runtime = context.get("execution_runtime") or select_backend(get_settings().data_processing)
    marker = Path(context["operation_directory"]) / "preparation.json"
    def configured_bundle() -> Path:
        return get_settings().paths.prepared_datasets_root / dataset_id / version_directory_name(uuid4())

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
    runtime_path = bundle / "execution_runtime.json"
    if runtime_path.is_file():
        data_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if any(data_runtime.get(key) != value for key, value in runtime.items()):
            raise ValueError("Prepared data runtime differs from the frozen run")
    else:
        component = ({"image_id": docker_image_id(get_settings().data_processing["image"])}
                      if runtime["backend"] == "docker" else {"packages": data_dependency_identity()})
        data_runtime = {**runtime, **component}
        atomic_write_json(runtime_path, data_runtime)
    for source in sources:
        if bundle.resolve().is_relative_to(Path(source.root).resolve()):
            raise ValueError("Configured prepared output overlaps a source directory")
        inventory_source(source, bundle / "source_manifests" / source.id)
    prepared = dict(context, sources=[source.model_dump() for source in sources], dataset_id=dataset_id,
                    dataset_scope=dataset_id, agent_role="DataProcessingAgent", execution_runtime=runtime,
                    data_runtime=data_runtime,
                    coding_worker_max_iterations=get_settings().data_processing["coding_worker_max_iterations"],
                    coding_worker_timeout_seconds=get_settings().data_processing["coding_worker_timeout_seconds"],
                    bundle_root=str(bundle), normalized_sources={}, source_scope=None,
                     source_roots={source.id: source.root for source in sources},
                     source_manifests={source.id: str(bundle / "source_manifests" / source.id) for source in sources})
    publish(prepared, "SourceSelection", {"sources": prepared["sources"]}, ())
    return prepared
