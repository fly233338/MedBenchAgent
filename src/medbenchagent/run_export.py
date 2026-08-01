"""Final run artifacts derived from the confirmed Blueprint and event log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medbenchagent.construction import construction_report, record_event
from medbenchagent.run_artifacts import atomic_write_json


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


def rag_designs_payload(
    blueprint: dict[str, Any], run_id: str, selected_material_files: list[str]
) -> dict[str, Any]:
    """Build the stable RAG-design artifact payload."""
    designs = {}
    fields = (
        "question_templates",
        "options",
        "answer_field",
        "answer_value_mapping",
        "generation_logic",
        "medical_task_description",
        "citations",
    )
    for task_id, task in blueprint["tasks"].items():
        if task.get("sample_count", 0) > 0:
            designs[task_id] = {field: task[field] for field in fields if field in task}
    return {
        "schema_version": "rag-designs.v2",
        "run_id": run_id,
        "selected_material_files": selected_material_files,
        "tasks": designs,
    }


def export_rag_designs(
    blueprint: dict[str, Any], run_id: str, selected_material_files: list[str], output_path: str | Path
) -> str:
    atomic_write_json(output_path, rag_designs_payload(blueprint, run_id, selected_material_files))
    return str(output_path)


def finalize_run(
    *,
    run_directory: str | Path,
    run_id: str,
    blueprint_path: str | Path,
    raw_output_path: str | Path,
    provenance_output_path: str | Path,
    rag_designs_path: str | Path,
    construction_events_path: str | Path,
    task_counts: dict[str, int],
    shortfalls: dict[str, int],
    dataset_catalog_entry: dict[str, str],
    dataset_profile: dict[str, Any],
    dataset_access_manifest: dict[str, Any],
    image_manifest_path: str | Path,
    audit_artifacts: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    run_dir = Path(run_directory)
    blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
    identifier = benchmark_id(blueprint)
    dataset_id = blueprint["dataset"]["dataset_id"]
    if dataset_catalog_entry.get("id") != dataset_id:
        raise ValueError("Dataset catalog entry does not match the confirmed Blueprint dataset ID.")
    image_manifest = Path(image_manifest_path)
    if not image_manifest.is_file():
        raise ValueError("image_manifest.jsonl is required to finalize a benchmark run.")
    record_event(
        construction_events_path,
        "run_completed",
        run_id=run_id,
        dataset_id=dataset_id,
        operation="finalize",
        component="orchestrator",
        status="complete",
    )
    report_path = run_dir / "construction_report.json"
    atomic_write_json(report_path, construction_report(construction_events_path))
    paths = {
        "blueprint": Path(blueprint_path),
        "raw": Path(raw_output_path),
        "provenance": Path(provenance_output_path),
        "rag_designs": Path(rag_designs_path),
        "construction_events": Path(construction_events_path),
        "construction_report": report_path,
    }
    effective_config = run_dir / "effective_config.json"
    if effective_config.is_file():
        paths["effective_config"] = effective_config
    if audit_artifacts:
        paths.update({name: Path(path) for name, path in audit_artifacts.items()})
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
