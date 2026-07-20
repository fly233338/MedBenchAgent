import json
from pathlib import Path

import pytest

import medbenchagent.generation as generation
from medbenchagent.dataset import prepare_annotation_dataset


def _record(index: int, annotation: dict) -> dict:
    return {
        "image_id": f"img-{index}",
        "image_path": f"images/{index}.png",
        "patient_id": None,
        "case_id": None,
        "study_id": None,
        "annotation": annotation,
    }


def _task(response_format: str, supporting_fields: list[str], **extra) -> dict:
    return {
        "task_type": "close_vqa" if response_format == "letter" else "open_vqa",
        "response_format": response_format,
        "description": "Test task",
        "require": "Approved annotation",
        "metric": "accuracy",
        "supporting_fields": supporting_fields,
        "dataset_support_reason": "Available.",
        "question_templates": ["Generate one question."],
        "generation_logic": "Use only the approved annotation.",
        "medical_task_description": "This assesses recognition; references support the terminology.",
        "citations": [],
        **extra,
    }


def _request(tmp_path: Path, records: list[dict], tasks: dict, target: int) -> dict:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    return {
        "benchmark_info": {"language": "en"},
        "dataset_access_manifest": manifest,
        "tasks": tasks,
        "target_sample_count": target,
        "seed": 0,
        "balance_answer_positions": True,
        "output_directory": str(tmp_path / "run"),
    }


def _generated(answer, field: str, question: str = "What is shown?", index=None) -> dict:
    return {
        "status": "generated",
        "question": question,
        "answer": answer,
        "source_refs": [{"field": field, "instance_index": index}],
    }


def test_generation_normalizes_letter_answer_and_retries_next_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"view": "craniocaudal"}), _record(1, {"view": "MLO"})],
        {"view": _task("letter", ["annotation.view"], options=["CC", "MLO"])},
        1,
    )
    generation_calls = 0

    def fake_complete_json(system: str, _prompt: str) -> dict:
        nonlocal generation_calls
        if system.startswith("Generate one"):
            generation_calls += 1
            return {} if generation_calls <= 3 else _generated("MLO", "annotation.view")
        return {"approved": True, "reason": "Faithful."}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    item = json.loads(Path(result.raw_output_path).read_text(encoding="utf-8"))

    assert item["image_ids"] == ["img-1"]
    assert item["options"] == ["CC", "MLO"]
    assert item["answer"] == "B"
    assert item["answer_value"] == "MLO"


def test_project_supporting_annotation_preserves_alignment_and_excludes_fields() -> None:
    annotation = {
        "findings": [
            {"kind": "mass", "bbox": [1, 2, 8, 9], "private": "omit"},
            {"kind": "calcification", "bbox": [10, 11, 20, 21], "private": "omit"},
        ],
        "report": "omit",
    }

    projected = generation.project_supporting_annotation(
        annotation,
        ["annotation.findings[*].kind", "annotation.findings[*].bbox[*]"],
    )

    assert projected == {
        "findings": [
            {"_source_index": 0, "kind": "mass", "bbox": [1, 2, 8, 9]},
            {"_source_index": 1, "kind": "calcification", "bbox": [10, 11, 20, 21]},
        ]
    }


def test_bbox_and_free_text_use_same_generate_and_review_flow(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [
            _record(0, {"finding": {"kind": "mass", "bbox": [1, 2, 8, 9]}, "report": "Expert report."}),
            _record(1, {"finding": {"kind": "mass", "bbox": [2, 3, 9, 10]}, "report": "Second report."}),
        ],
        {
            "detection": _task(
                "bbox", ["annotation.finding.kind", "annotation.finding.bbox[*]"]
            ),
            "report": _task("free_text", ["annotation.report"]),
        },
        2,
    )

    def fake_complete_json(system: str, prompt: str) -> dict:
        if system.startswith("Independently"):
            return {"approved": True, "reason": "Faithful."}
        if "Task ID: detection" in prompt:
            return _generated([1, 2, 8, 9], "annotation.finding.bbox[*]", "Locate the mass.")
        return _generated("Expert report.", "annotation.report", "Generate a report.")

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    items = [
        json.loads(line)
        for line in Path(result.raw_output_path).read_text(encoding="utf-8").splitlines()
    ]

    assert result.task_counts == {"detection": 1, "report": 1}
    assert items[0]["answer"] == [1.0, 2.0, 8.0, 9.0]
    assert items[1]["answer"] == "Expert report."


def test_generation_rejects_unreviewed_items_and_fails_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign"})],
        {"status": _task("letter", ["annotation.status"], options=["benign", "malignant"])},
        1,
    )
    monkeypatch.setattr(
        generation,
        "complete_json",
        lambda system, _prompt: _generated("benign", "annotation.status")
        if system.startswith("Generate one")
        else {"approved": False, "reason": "Leaky."},
    )

    with pytest.raises(ValueError, match="Unable to generate"):
        generation.generate_raw_benchmark(request)
    assert not (Path(request["output_directory"]) / "benchmark_items.raw.jsonl").exists()


def test_generation_fails_before_output_when_candidates_are_insufficient(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign"})],
        {"status": _task("letter", ["annotation.status"], options=["benign", "malignant"])},
        2,
    )

    with pytest.raises(ValueError, match="Insufficient valid candidates"):
        generation.generate_raw_benchmark(request)
