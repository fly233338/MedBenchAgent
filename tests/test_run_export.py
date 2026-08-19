import hashlib
import json
from pathlib import Path

import pytest

from medbenchagent.dataset import CACHE_FORMAT_VERSION, PROFILE_SCHEMA_VERSION
from medbenchagent.run_artifacts import write_versioned_json
from medbenchagent.run_export import (
    _image_manifest_format,
    benchmark_id,
    finalize_run,
    rag_designs_payload,
)


def _blueprint() -> dict:
    return {
        "benchmark_info": {"disease": "breast cancer", "sample_count": 1},
        "dataset": {
            "dataset_id": "vindr-mammo",
        },
        "tasks": {
            "view": {
                "parent_task_id": "image_acquisition_attribute",
                "question_type_id": "view",
                "category": "image_acquisition_and_basic_information",
                "response_format": "letter",
                "supporting_fields": ["image.annotation.view"],
                "sample_count": 1,
                "metric": "accuracy",
                "question_templates": ["Which view?"],
                "generation_logic": "Use supporting fields.",
                "medical_task_description": "Recognition.",
                "citations": [],
            },
            "zero": {
                "category": "report_generation",
                "response_format": "free_text",
                "supporting_fields": ["image.annotation.report"],
                "sample_count": 0,
                "metric": "bleu",
            },
        },
    }


def test_image_manifest_format_detects_original_jpeg(tmp_path: Path) -> None:
    manifest = tmp_path / "image_manifest.jsonl"
    manifest.write_text(
        '{"image_id":"image-one","png_path":"images/image-one.jpg",'
        '"width":20,"height":10,"sha256":"' + "a" * 64 + '"}\n',
        encoding="utf-8",
    )

    assert _image_manifest_format(manifest) == "original_jpeg"


def _audit_artifacts(run_directory: Path) -> dict[str, str]:
    paths = {
        "audited": run_directory / "benchmark.vindr-mammo.audited.jsonl",
        "flagged": run_directory / "benchmark.vindr-mammo.flagged.jsonl",
        "rejected": run_directory / "benchmark.vindr-mammo.rejected.jsonl",
        "records": run_directory / "audit_records.jsonl",
        "report": run_directory / "audit_report.json",
        "semantic_records": run_directory / "semantic_audit_records.jsonl",
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _finalize_inputs(tmp_path: Path) -> dict:
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(_blueprint()), encoding="utf-8")
    raw = tmp_path / "benchmark.vindr-mammo.raw.jsonl"
    provenance = tmp_path / "item_provenance.jsonl"
    events = tmp_path / "construction_events.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    provenance.write_text("{}\n", encoding="utf-8")
    events.touch()
    (tmp_path / "effective_config.json").write_text(
        '{"settings": {}}\n', encoding="utf-8"
    )
    image_manifest = tmp_path / "image_manifest.jsonl"
    image_manifest.write_text("{}\n", encoding="utf-8")
    rag_artifact = write_versioned_json(
        tmp_path,
        "rag_designs",
        rag_designs_payload(_blueprint(), "run-1", ["guide.pdf"]),
    )
    return {
        "run_directory": tmp_path,
        "run_id": "run-1",
        "blueprint_path": blueprint_path,
        "raw_output_path": raw,
        "provenance_output_path": provenance,
        "rag_designs_path": str(rag_artifact["path"]),
        "construction_events_path": events,
        "dataset_catalog_entry": {
            "id": "vindr-mammo",
            "name": "VinDr-Mammo",
            "description": "Mammography.",
        },
        "dataset_profile": {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "total_images": 1,
        },
        "dataset_access_manifest": {
            "annotation_sha256": "a" * 64,
            "cache_format_version": CACHE_FORMAT_VERSION,
        },
        "image_manifest_path": image_manifest,
        "audit_artifacts": _audit_artifacts(tmp_path),
    }


def test_exports_confirmed_rag_designs_and_content_addressed_manifest(tmp_path: Path) -> None:
    inputs = _finalize_inputs(tmp_path)
    raw = Path(inputs["raw_output_path"])
    events = Path(inputs["construction_events_path"])
    image_manifest = Path(inputs["image_manifest_path"])
    effective_config = tmp_path / "effective_config.json"
    rag = str(inputs["rag_designs_path"])

    identifier, report, manifest = finalize_run(**inputs)

    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    assert identifier == benchmark_id(_blueprint())
    assert payload["tasks"]["zero"]["metrics"] == ["bleu"]
    assert payload["tasks"]["view"]["parent_task_id"] == "image_acquisition_attribute"
    assert payload["tasks"]["view"]["question_type_id"] == "view"
    assert payload["dataset"] == {
        "dataset_id": "vindr-mammo",
        "dataset_name": "VinDr-Mammo",
        "description": "Mammography.",
        "annotation_sha256": "a" * 64,
        "image_manifest_sha256": hashlib.sha256(image_manifest.read_bytes()).hexdigest(),
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "cache_schema_version": CACHE_FORMAT_VERSION,
        "total_images": 1,
        "image_format": "8bit_grayscale_png",
    }
    assert payload["artifacts"]["raw"]["path"] == raw.name
    assert payload["artifacts"]["rag_designs"]["path"] == str(Path("rag_designs") / "v1.json")
    assert payload["artifacts"]["effective_config"] == {
        "path": effective_config.name,
        "sha256": hashlib.sha256(effective_config.read_bytes()).hexdigest(),
    }
    assert payload["artifacts"]["manifest"] == {"path": "benchmark_manifest.json"}
    rag_payload = json.loads(Path(rag).read_text(encoding="utf-8"))
    assert rag_payload["schema_version"] == "rag-designs.v2"
    assert rag_payload["tasks"] == {
        "view": {
            "question_templates": ["Which view?"],
            "generation_logic": "Use supporting fields.",
            "medical_task_description": "Recognition.",
            "citations": [],
        }
    }
    assert Path(report).is_file()
    completed = [
        json.loads(line)
        for line in events.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "run_completed"
    ]
    assert completed == [
        {
            "timestamp": completed[0]["timestamp"],
            "event_type": "run_completed",
            "run_id": "run-1",
            "dataset_id": "vindr-mammo",
            "operation": "finalize",
            "component": "orchestrator",
            "status": "complete",
        }
    ]


def test_finalize_rejects_blueprint_without_metric(tmp_path: Path) -> None:
    blueprint = _blueprint()
    blueprint["tasks"]["view"].pop("metric")
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")
    image_manifest = tmp_path / "image_manifest.jsonl"
    image_manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="view: final Blueprint task must contain a metric"):
        finalize_run(
            run_directory=tmp_path,
            run_id="run-1",
            blueprint_path=blueprint_path,
            raw_output_path=tmp_path / "raw.jsonl",
            provenance_output_path=tmp_path / "provenance.jsonl",
            rag_designs_path=tmp_path / "rag_designs.json",
            construction_events_path=tmp_path / "construction_events.jsonl",
            dataset_catalog_entry={
                "id": "vindr-mammo",
                "name": "VinDr-Mammo",
                "description": "Mammography.",
            },
            dataset_profile={},
            dataset_access_manifest={},
            image_manifest_path=image_manifest,
            audit_artifacts={},
        )


@pytest.mark.parametrize("missing", ["audit", "config"])
def test_finalize_does_not_record_completion_when_required_artifacts_are_missing(
    tmp_path: Path, missing: str
) -> None:
    inputs = _finalize_inputs(tmp_path)
    events = Path(inputs["construction_events_path"])
    if missing == "audit":
        inputs["audit_artifacts"].pop("semantic_records")
        expected = "missing required files"
    else:
        (tmp_path / "effective_config.json").unlink()
        expected = "effective_config"

    with pytest.raises(ValueError, match=expected):
        finalize_run(**inputs)

    assert all(
        json.loads(line).get("event_type") != "run_completed"
        for line in events.read_text(encoding="utf-8").splitlines()
    )
