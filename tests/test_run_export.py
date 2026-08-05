import hashlib
import json
from pathlib import Path

import pytest

from medbenchagent.run_export import benchmark_id, export_rag_designs, finalize_run


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


def test_exports_confirmed_rag_designs_and_content_addressed_manifest(tmp_path: Path) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(_blueprint()), encoding="utf-8")
    raw = tmp_path / "benchmark.vindr-mammo.raw.jsonl"
    provenance = tmp_path / "item_provenance.jsonl"
    events = tmp_path / "construction_events.jsonl"
    effective_config = tmp_path / "effective_config.json"
    raw.write_text("{}\n", encoding="utf-8")
    provenance.write_text("{}\n", encoding="utf-8")
    events.touch()
    effective_config.write_text('{"settings": {}}\n', encoding="utf-8")
    image_manifest = tmp_path / "image_manifest.jsonl"
    image_manifest.write_text("{}\n", encoding="utf-8")
    rag = export_rag_designs(
        _blueprint(), "run-1", ["guide.pdf"], tmp_path / "rag_designs" / "v2.json"
    )

    identifier, report, manifest = finalize_run(
        run_directory=tmp_path, run_id="run-1", blueprint_path=blueprint_path, raw_output_path=raw,
        provenance_output_path=provenance, rag_designs_path=rag, construction_events_path=events,
        task_counts={"view": 1, "zero": 0}, shortfalls={"view": 0, "zero": 0},
        dataset_catalog_entry={"id": "vindr-mammo", "name": "VinDr-Mammo", "description": "Mammography."},
        dataset_profile={"schema_version": "profile.v1", "total_images": 1},
        dataset_access_manifest={"annotation_sha256": "a" * 64, "cache_format_version": "cache.v1"},
        image_manifest_path=image_manifest,
    )

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
        "profile_schema_version": "profile.v1",
        "cache_schema_version": "cache.v1",
        "total_images": 1,
        "image_format": "8bit_grayscale_png",
    }
    assert payload["artifacts"]["raw"]["path"] == raw.name
    assert payload["artifacts"]["rag_designs"]["path"] == str(Path("rag_designs") / "v2.json")
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
            task_counts={},
            shortfalls={},
            dataset_catalog_entry={
                "id": "vindr-mammo",
                "name": "VinDr-Mammo",
                "description": "Mammography.",
            },
            dataset_profile={},
            dataset_access_manifest={},
            image_manifest_path=image_manifest,
        )
