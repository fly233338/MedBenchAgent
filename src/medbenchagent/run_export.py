"""Final run artifacts derived from the confirmed Blueprint and event log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from medbenchagent.construction import construction_report, record_event
from medbenchagent.run_artifacts import atomic_write_json


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def benchmark_id(blueprint: dict[str, Any]) -> str:
    """Content-address the final public benchmark specification, never its run ID."""
    return "benchmark-" + _canonical_hash(
        {
            "benchmark_info": blueprint["benchmark_info"],
            "dataset": blueprint["dataset"],
            "tasks": blueprint["tasks"],
        }
    )[:16]


def export_rag_designs(
    blueprint: dict[str, Any], run_id: str, selected_material_files: list[str], output_path: str | Path
) -> str:
    designs = {}
    fields = ("question_templates", "options", "generation_logic", "medical_task_description", "citations")
    for task_id, task in blueprint["tasks"].items():
        if task.get("sample_count", 0) > 0:
            designs[task_id] = {field: task[field] for field in fields if field in task}
    atomic_write_json(
        output_path,
        {
            "schema_version": "rag-designs.v1",
            "run_id": run_id,
            "selected_material_files": selected_material_files,
            "tasks": designs,
        },
    )
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
) -> tuple[str, str, str]:
    run_dir = Path(run_directory)
    blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
    identifier = benchmark_id(blueprint)
    record_event(
        construction_events_path,
        "generation_summary",
        run_id=run_id,
        dataset_id=blueprint["dataset"]["dataset_id"],
        agent="generation",
        operation="summary",
        status="complete",
        task_counts=task_counts,
        shortfalls=shortfalls,
    )
    record_event(
        construction_events_path,
        "run_completed",
        run_id=run_id,
        dataset_id=blueprint["dataset"]["dataset_id"],
        agent="workflow",
        operation="finalize",
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
    manifest_path = run_dir / "benchmark_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "benchmark-manifest.v1",
            "benchmark_id": identifier,
            "run_id": run_id,
            "benchmark_info": blueprint["benchmark_info"],
            "dataset": blueprint["dataset"],
            "tasks": {
                task_id: {
                    "category": task["category"],
                    "response_format": task["response_format"],
                    "supporting_fields": task["supporting_fields"],
                    "sample_count": task["sample_count"],
                    "metrics": [task["metric"]],
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
