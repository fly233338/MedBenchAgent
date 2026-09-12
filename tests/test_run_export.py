import json
from pathlib import Path
import pytest
from medbenchagent.run_export import finalize_run


def inputs(tmp_path):
    blueprint = {
        "benchmark_info": {"language": "en"},
        "dataset": {"dataset_id": "set"},
        "tasks": {
            "view": {
                "category": "basic",
                "response_format": "letter",
                "metric": "accuracy",
                "supporting_fields": ["image.annotation.view"],
                "sample_count": 1,
            }
        },
    }
    for name, value in {
        "blueprint.json": blueprint,
        "effective_config.json": {},
        "audit_report.json": {},
        "image_manifest.jsonl": {},
    }.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    for name in ("raw.jsonl", "audited.jsonl", "construction_events.jsonl"):
        (tmp_path / name).touch()
    return dict(
        run_directory=tmp_path,
        run_id="run",
        blueprint_path=tmp_path / "blueprint.json",
        raw_output_path=tmp_path / "raw.jsonl",
        construction_events_path=tmp_path / "construction_events.jsonl",
        dataset_catalog_entry={"id": "set", "name": "Set", "description": "Test"},
        dataset_profile={"schema_version": "v3", "total_images": 1},
        dataset_access_manifest={"cache_format_version": "v7"},
        image_manifest_path=tmp_path / "image_manifest.jsonl",
        audit_artifacts={
            "audited": str(tmp_path / "audited.jsonl"),
            "report": str(tmp_path / "audit_report.json"),
        },
    )


def test_final_manifest_uses_run_identity_and_relative_paths(tmp_path):
    identifier, path = finalize_run(**inputs(tmp_path))
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert identifier == "benchmark-run"
    assert manifest["schema_version"] == "benchmark-manifest.v2"
    assert manifest["artifacts"]["raw"] == {"path": "raw.jsonl"}
    assert manifest["tasks"]["view"]["metrics"] == ["accuracy"]


def test_missing_audit_cannot_complete(tmp_path):
    values = inputs(tmp_path)
    values["audit_artifacts"].pop("report")
    with pytest.raises(ValueError, match="missing required"):
        finalize_run(**values)
    assert not (tmp_path / "construction_events.jsonl").read_text()


def test_final_task_requires_metric(tmp_path):
    values = inputs(tmp_path)
    path = values["blueprint_path"]
    blueprint = json.loads(path.read_text())
    blueprint["tasks"]["view"].pop("metric")
    path.write_text(json.dumps(blueprint))
    with pytest.raises(ValueError, match="metric"):
        finalize_run(**values)
