"""Final run artifacts derived from the confirmed Blueprint and event log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medbenchagent.contracts import RAG_DESIGN_FIELDS
from medbenchagent.construction import construction_report, record_event
from medbenchagent.metrics import is_metric_supported
from medbenchagent.run_artifacts import atomic_write_json

_REQUIRED_AUDIT_ARTIFACTS = frozenset(
    {"audited", "flagged", "rejected", "records", "report", "semantic_records"}
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def benchmark_id(blueprint: dict[str, Any]) -> str:
    """Content-address the final public benchmark specification, never its run ID."""
    return "benchmark-" + _canonical_hash(
        {
            "benchmark_info": blueprint["benchmark_info"],
            "dataset_id": blueprint["dataset"]["dataset_id"],
            "tasks": blueprint["tasks"],
        }
    )[:16]


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


def rag_designs_payload(
    blueprint: dict[str, Any], run_id: str, selected_material_files: list[str]
) -> dict[str, Any]:
    """Build the stable RAG-design artifact payload."""
    designs = {}
    for task_id, task in blueprint["tasks"].items():
        if task.get("sample_count", 0) > 0:
            designs[task_id] = {
                field: task[field] for field in RAG_DESIGN_FIELDS if field in task
            }
    return {
        "schema_version": "rag-designs.v2",
        "run_id": run_id,
        "selected_material_files": selected_material_files,
        "tasks": designs,
    }


def finalize_run(
    *,
    run_directory: str | Path,
    run_id: str,
    blueprint_path: str | Path,
    raw_output_path: str | Path,
    provenance_output_path: str | Path,
    rag_designs_path: str | Path,
    construction_events_path: str | Path,
    dataset_catalog_entry: dict[str, str],
    dataset_profile: dict[str, Any],
    dataset_access_manifest: dict[str, Any],
    image_manifest_path: str | Path,
    audit_artifacts: dict[str, str],
) -> tuple[str, str, str]:
    run_dir = Path(run_directory).resolve()
    blueprint_file = _required_run_file(blueprint_path, "blueprint", run_dir)
    blueprint = json.loads(blueprint_file.read_text(encoding="utf-8"))
    identifier = benchmark_id(blueprint)
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
        "provenance": _required_run_file(
            provenance_output_path, "provenance", run_dir
        ),
        "rag_designs": _required_run_file(rag_designs_path, "rag_designs", run_dir),
        "construction_events": _required_run_file(
            construction_events_path, "construction_events", run_dir
        ),
        "effective_config": _required_run_file(
            run_dir / "effective_config.json", "effective_config", run_dir
        ),
    }
    for name, path in audit_artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("audit_artifacts keys must be non-empty strings.")
        if name in paths or name == "construction_report":
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
    report_path = run_dir / "construction_report.json"
    atomic_write_json(report_path, construction_report(paths["construction_events"]))
    paths["construction_report"] = report_path
    manifest_path = run_dir / "benchmark_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "benchmark-manifest.v1",
            "benchmark_id": identifier,
            "run_id": run_id,
            "benchmark_info": blueprint["benchmark_info"],
            "dataset": {
                "dataset_id": dataset_id,
                "dataset_name": dataset_catalog_entry["name"],
                "description": dataset_catalog_entry["description"],
                "annotation_sha256": dataset_access_manifest["annotation_sha256"],
                "image_manifest_sha256": _sha256(image_manifest),
                "profile_schema_version": dataset_profile["schema_version"],
                "cache_schema_version": dataset_access_manifest["cache_format_version"],
                "total_images": dataset_profile["total_images"],
                "image_format": "8bit_grayscale_png",
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
                name: {"path": str(path.relative_to(run_dir)), "sha256": _sha256(path)}
                for name, path in paths.items()
            }
            | {"manifest": {"path": manifest_path.name}},
        },
    )
    return identifier, str(report_path), str(manifest_path)
