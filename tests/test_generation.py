import json
from pathlib import Path

import pytest

import medbenchagent.generation as generation
from medbenchagent.dataset import prepare_annotation_dataset


def _record(index: int, value: str) -> dict:
    return {
        "image_id": f"img-{index}", "image_path": f"images/{index}.png", "patient_id": None,
        "case_id": None, "study_id": None, "annotation": {"status": value},
    }


def _request(tmp_path: Path, values: list[str], target: int) -> dict:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text("".join(json.dumps(_record(index, value)) + "\n" for index, value in enumerate(values)), encoding="utf-8")
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    return {
        "benchmark_info": {}, "dataset_access_manifest": manifest, "approved_task_plans": [],
        "question_designs": [],
        "task_specs": [{
            "task_id": "status", "task_type": "close_vqa", "response_format": "letter",
            "description": "Status", "question_templates": ["What is the status?"],
            "placeholder_bindings": {}, "validation_rules": [], "citations": [],
            "answer_field": "annotation.status", "answer_value_mapping": {},
            "options": ["benign", "malignant"],
        }],
        "target_sample_count": target, "seed": 0, "balance_answer_positions": True, "output_directory": str(tmp_path / "run"),
    }


def test_generation_keeps_options_and_retries_then_uses_next_candidate(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path, ["benign", "malignant"], 1)
    calls = 0

    def fake_complete_json(system: str, _prompt: str) -> dict:
        nonlocal calls
        if system.startswith("You write"):
            calls += 1
            return {} if calls <= 3 else {"question": "What is the image-associated status?"}
        return {"approved": True}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    item = json.loads(Path(result.raw_output_path).read_text(encoding="utf-8"))

    assert result.item_count == 1
    assert item["image_ids"] == ["img-1"]
    assert item["options"] == ["benign", "malignant"]
    assert item["answer"] == "B"
    assert item["answer_value"] == "malignant"


def test_generation_fails_before_output_when_candidates_are_insufficient(tmp_path: Path) -> None:
    request = _request(tmp_path, ["benign", "malignant"], 3)

    with pytest.raises(ValueError, match="Insufficient valid candidates"):
        generation.generate_raw_benchmark(request)
    assert not (Path(request["output_directory"]) / "benchmark_items.raw.jsonl").exists()


def test_detection_generates_one_aligned_item_per_valid_finding(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    record = {
        "image_id": "img-1", "image_path": "images/1.png", "patient_id": None,
        "case_id": None, "study_id": None,
        "annotation": {
            "findings": [
                {"kind": "mass", "bbox": [1, 2, 8, 9]},
                {"kind": "calcification", "bbox": [10, 11, 20, 21]},
                {"kind": "invalid", "bbox": [5, 5, 4, 4]},
            ]
        },
    }
    annotation_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    request = {
        "benchmark_info": {}, "dataset_access_manifest": manifest, "approved_task_plans": [],
        "question_designs": [],
        "task_specs": [{
            "task_id": "detection", "task_type": "open_vqa", "response_format": "bbox",
            "description": "Locate a finding", "question_templates": ["Locate the {target_text}."],
            "placeholder_bindings": {"target_text": "annotation.findings[*].kind"},
            "validation_rules": [], "citations": [], "instance_path": "annotation.findings[*]",
            "target_text_field": "annotation.findings[*].kind",
            "bbox_field": "annotation.findings[*].bbox[*]", "bbox_format": "xyxy",
        }],
        "target_sample_count": 2, "seed": 7, "balance_answer_positions": True,
        "output_directory": str(tmp_path / "run"),
    }

    result = generation.generate_raw_benchmark(request)
    items = [json.loads(line) for line in Path(result.raw_output_path).read_text(encoding="utf-8").splitlines()]

    assert [item["question"] for item in items] == ["Locate the mass.", "Locate the calcification."]
    assert [item["answer"] for item in items] == [[1.0, 2.0, 8.0, 9.0], [10.0, 11.0, 20.0, 21.0]]
    assert len({item["id"] for item in items}) == 2
    assert all(item["options"] is None and "answer_value" not in item for item in items)


def test_free_text_tasks_use_only_paired_expert_text(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    records = [
        _record(0, "benign") | {"annotation": {"report": "Expert report."}},
        _record(1, "benign") | {"annotation": {"report": 123}},
    ]
    annotation_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    request = {
        "benchmark_info": {}, "dataset_access_manifest": manifest, "approved_task_plans": [],
        "question_designs": [],
        "task_specs": [{
            "task_id": "report_generation", "task_type": "open_vqa", "response_format": "free_text",
            "description": "Generate a report", "question_templates": ["Generate a report for this image."],
            "placeholder_bindings": {}, "validation_rules": [], "citations": [],
            "answer_field": "annotation.report",
        }],
        "target_sample_count": 1, "seed": 0, "balance_answer_positions": True,
        "output_directory": str(tmp_path / "run"),
    }

    result = generation.generate_raw_benchmark(request)
    item = json.loads(Path(result.raw_output_path).read_text(encoding="utf-8"))

    assert item["question"] == "Generate a report for this image."
    assert item["answer"] == "Expert report."
    assert item["response_format"] == "free_text"


def test_mixed_tasks_share_target_evenly(tmp_path: Path, monkeypatch) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    records = [
        _record(0, "benign") | {"annotation": {"status": "benign", "report": "First report."}},
        _record(1, "malignant") | {"annotation": {"status": "malignant", "report": "Second report."}},
    ]
    annotation_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    request = {
        "benchmark_info": {}, "dataset_access_manifest": manifest, "approved_task_plans": [],
        "question_designs": [],
        "task_specs": [
            {
                "task_id": "status", "task_type": "close_vqa", "response_format": "letter",
                "description": "Status", "question_templates": ["What is the status?"],
                "placeholder_bindings": {}, "validation_rules": [], "citations": [],
                "answer_field": "annotation.status", "answer_value_mapping": {},
                "options": ["benign", "malignant"],
            },
            {
                "task_id": "report_generation", "task_type": "open_vqa", "response_format": "free_text",
                "description": "Report", "question_templates": ["Generate a report."],
                "placeholder_bindings": {}, "validation_rules": [], "citations": [],
                "answer_field": "annotation.report",
            },
        ],
        "target_sample_count": 2, "seed": 0, "balance_answer_positions": True,
        "output_directory": str(tmp_path / "run"),
    }
    monkeypatch.setattr(
        generation,
        "complete_json",
        lambda system, _prompt: {"question": "What is the status?"}
        if system.startswith("You write")
        else {"approved": True},
    )

    result = generation.generate_raw_benchmark(request)
    items = [json.loads(line) for line in Path(result.raw_output_path).read_text(encoding="utf-8").splitlines()]

    assert result.task_counts == {"status": 1, "report_generation": 1}
    assert [item["response_format"] for item in items] == ["letter", "free_text"]
