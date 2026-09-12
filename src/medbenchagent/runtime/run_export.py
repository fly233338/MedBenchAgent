"""Final run artifacts derived from the confirmed Blueprint and event log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medbenchagent.runtime.construction import record_event
from medbenchagent.metrics import is_metric_supported
from medbenchagent.runtime.run_artifacts import atomic_write_json

_REQUIRED_AUDIT_ARTIFACTS = frozenset(
    {"audited", "report"}
)








def _required_run_file(path: str | Path, name: str, run_directory: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(run_directory)
    except ValueError as exc:
        raise ValueError(f"{name} must be stored inside the run directory.") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing regular file.")
    try:
        with resolved.open("rb") as source:
            source.read(1)
    except OSError as exc:
        raise ValueError(f"{name} must be readable.") from exc
    return resolved


def _required_readable_file(path: str | Path, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing regular file.")
    try:
        with resolved.open("rb") as source:
            source.read(1)
    except OSError as exc:
        raise ValueError(f"{name} must be readable.") from exc
    return resolved


def _image_manifest_format(path: Path) -> str:
    """Return a stable format label while keeping the legacy manifest shape."""
    suffixes: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            relative = row.get("png_path") if isinstance(row, dict) else None
            if isinstance(relative, str):
                suffixes.add(Path(relative).suffix.lower())
    except (OSError, json.JSONDecodeError):
        return "unknown_raster_image"
    if not suffixes or suffixes == {".png"}:
        return "png"
    if suffixes <= {".jpg", ".jpeg"}:
        return "original_jpeg"
    return "mixed_raster_images"


def finalize_run(
    *,
    run_directory: str | Path,
    run_id: str,
    blueprint_path: str | Path,
    raw_output_path: str | Path,
    construction_events_path: str | Path,
    dataset_catalog_entry: dict[str, str],
    dataset_profile: dict[str, Any],
    dataset_access_manifest: dict[str, Any],
    image_manifest_path: str | Path,
    audit_artifacts: dict[str, str],
) -> tuple[str, str]:
    run_dir = Path(run_directory).resolve()
    blueprint_file = _required_run_file(blueprint_path, "blueprint", run_dir)
    blueprint = json.loads(blueprint_file.read_text(encoding="utf-8"))
    identifier = "benchmark-" + run_id
    dataset_id = blueprint["dataset"]["dataset_id"]
    if dataset_catalog_entry.get("id") != dataset_id:
        raise ValueError("Dataset catalog entry does not match the confirmed Blueprint dataset ID.")
    image_manifest = _required_readable_file(image_manifest_path, "image_manifest.jsonl")
    for task_id, task in blueprint["tasks"].items():
        if not is_metric_supported(task.get("response_format"), task.get("metric")):
            raise ValueError(
                f"{task_id}: final Blueprint task must contain a metric compatible with "
                f"response format {task.get('response_format')}."
            )
    if not isinstance(audit_artifacts, dict):
        raise ValueError("audit_artifacts must be a dictionary.")
    missing_audit_artifacts = _REQUIRED_AUDIT_ARTIFACTS - set(audit_artifacts)
    if missing_audit_artifacts:
        raise ValueError(
            "audit_artifacts is missing required files: "
            + ", ".join(sorted(missing_audit_artifacts))
        )
    paths = {
        "blueprint": blueprint_file,
        "raw": _required_run_file(raw_output_path, "raw", run_dir),
        "construction_events": _required_run_file(
            construction_events_path, "construction_events", run_dir
        ),
        "effective_config": _required_run_file(
            run_dir / "effective_config.json", "effective_config", run_dir
        ),
    }
    if (run_dir / "templates.json").is_file():
        paths["templates"] = run_dir / "templates.json"
    for name, path in audit_artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("audit_artifacts keys must be non-empty strings.")
        if name in paths:
            raise ValueError(f"audit artifact name is reserved: {name}")
        paths[name] = _required_run_file(path, f"audit artifact {name}", run_dir)
    record_event(
        paths["construction_events"],
        "run_completed",
        run_id=run_id,
        dataset_id=dataset_id,
        operation="finalize",
        component="orchestrator",
        status="complete",
    )
    manifest_path = run_dir / "benchmark_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "benchmark-manifest.v2",
            "benchmark_id": identifier,
            "run_id": run_id,
            "benchmark_info": blueprint["benchmark_info"],
            "dataset": {
                "dataset_id": dataset_id,
                "dataset_name": dataset_catalog_entry["name"],
                "description": dataset_catalog_entry["description"],
                "profile_schema_version": dataset_profile["schema_version"],
                "cache_schema_version": dataset_access_manifest["cache_format_version"],
                "total_images": dataset_profile["total_images"],
                "image_format": _image_manifest_format(image_manifest),
            },
            "tasks": {
                task_id: {
                    "category": task["category"],
                    "response_format": task["response_format"],
                    "supporting_fields": task["supporting_fields"],
                    "sample_count": task["sample_count"],
                    "metrics": [task["metric"]],
                    **(
                        {"parent_task_id": task["parent_task_id"]}
                        if isinstance(task.get("parent_task_id"), str)
                        else {}
                    ),
                    **(
                        {"question_type_id": task["question_type_id"]}
                        if isinstance(task.get("question_type_id"), str)
                        else {}
                    ),
                }
                for task_id, task in blueprint["tasks"].items()
            },
            "artifacts": {
                name: {"path": str(path.relative_to(run_dir))}
                for name, path in paths.items()
            }
            | {"manifest": {"path": manifest_path.name}},
        },
    )
    return identifier, str(manifest_path)
