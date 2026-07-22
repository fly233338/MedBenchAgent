import json
from pathlib import Path

import pytest

import medbenchagent.generation as generation
from medbenchagent.dataset import prepare_annotation_dataset


def _record(index: int, annotation: dict) -> dict:
    return {
        "image_id": f"img-{index}",
        "image_path": f"images/{index}.dicom",
        "patient_id": None,
        "case_id": None,
        "study_id": None,
        "annotation": annotation,
    }


def _task(field: str, **extra) -> dict:
    return {
        "task_type": "close_vqa",
        "response_format": "letter",
        "description": "Identify the status",
        "require": "Use the approved annotation",
        "metric": "accuracy",
        "supporting_fields": [field],
        "dataset_support_reason": "Available.",
        "question_templates": ["What is the status shown in this image?"],
        "generation_logic": "Normalize only the approved annotation.",
        "medical_task_description": "Recognition task.",
        "citations": [],
        "options": ["benign", "malignant"],
        **extra,
    }


def _request(tmp_path: Path, records: list[dict], tasks: dict, target: int, seed: int = 0) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    return {
        "benchmark_info": {"language": "en", "modality": "mammography"},
        "dataset_access_manifest": manifest,
        "tasks": tasks,
        "target_sample_count": target,
        "seed": seed,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
        "output_directory": str(tmp_path / "run"),
    }


def _read_items(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_letter_generation_uses_single_task_prompt_temperature_and_compact_raw(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign"})],
        {"status": _task("annotation.status", options=["A. benign", "B) malignant"])},
        1,
    )
    calls: list[tuple[str, str, float]] = []

    def fake_complete_json(system: str, user: str, *, temperature: float = 0) -> dict:
        calls.append((system, user, temperature))
        if system.startswith("Review one"):
            return {"approved": True, "reason": "Faithful."}
        return {"question": "Which status best describes this image?", "answer": "benign"}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    item = _read_items(result.raw_output_path)[0]

    prompt, user, temperature = calls[0]
    assert "Task ID: status" in prompt
    assert "Candidate options: [\"benign\", \"malignant\"]" in prompt
    assert "Supporting annotation: {\"status\": \"benign\"}" in prompt
    assert "citations" not in prompt.lower()
    assert user == ""
    assert temperature == 0.7
    assert calls[1][2] == 0
    assert item == {
        "id": item["id"],
        "task_id": "status",
        "image_id": "img-0",
        "image_path": "images/0.dicom",
        "question": "Which status best describes this image?",
        "options": ["benign", "malignant"],
        "answer": "A",
    }


def test_rag_and_generation_require_one_template_and_clean_options() -> None:
    task = _task("annotation.status", question_templates=["one", "two"])
    request = {
        "tasks": {"status": task},
        "target_sample_count": 1,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
    }
    with pytest.raises(ValueError, match="exactly one"):
        generation._validate_tasks(request)

    task["question_templates"] = ["one"]
    task["options"] = ["A. benign", "B) malignant"]
    generation._validate_tasks(request)
    assert task["options"] == ["benign", "malignant"]


def test_supporting_annotation_requires_meaningful_values() -> None:
    assert not generation._has_meaningful_value({"status": None})
    assert not generation._has_meaningful_value({"status": ""})
    assert not generation._has_meaningful_value({"findings": []})
    assert not generation._has_meaningful_value({"finding": {}})
    assert generation._has_meaningful_value({"status": 0})
    assert generation._has_meaningful_value({"status": False})


def test_seeded_candidate_order_is_stable_and_varies_with_seed(tmp_path: Path) -> None:
    records = [_record(index, {"a": "benign", "b": "benign", "c": "benign"}) for index in range(5)]
    tasks = {task_id: _task(f"annotation.{task_id}") for task_id in ("a", "b", "c")}
    zero = _request(tmp_path / "zero", records, tasks, 3, seed=0)
    one = _request(tmp_path / "one", records, tasks, 3, seed=1)
    zero_tasks, zero_targets = generation._validate_tasks(zero)
    one_tasks, one_targets = generation._validate_tasks(one)

    _, first_zero = generation._next_candidate_record(zero, zero_tasks, zero_targets, {key: 0 for key in tasks}, 0)
    _, second_zero = generation._next_candidate_record(zero, zero_tasks, zero_targets, {key: 0 for key in tasks}, 0)
    _, first_one = generation._next_candidate_record(one, one_tasks, one_targets, {key: 0 for key in tasks}, 0)

    assert [candidate.task_id for candidate in first_zero] == [candidate.task_id for candidate in second_zero]
    assert [candidate.task_id for candidate in first_zero] != [candidate.task_id for candidate in first_one]


def test_completed_tasks_do_not_enter_candidate_queue(tmp_path: Path) -> None:
    records = [_record(0, {"a": "benign", "b": "benign"})]
    tasks = {task_id: _task(f"annotation.{task_id}", sample_count=1) for task_id in ("a", "b")}
    request = _request(tmp_path, records, tasks, 2)
    validated, targets = generation._validate_tasks(request)

    _, candidates = generation._next_candidate_record(request, validated, targets, {"a": 1, "b": 0}, 0)

    assert [candidate.task_id for candidate in candidates] == ["b"]


def test_failed_or_rejected_candidate_is_replaced_before_image_limit(tmp_path: Path, monkeypatch) -> None:
    fields = ("a", "b", "c", "failed")
    tasks = {
        field: _task(f"annotation.{field}", description=field, sample_count=1)
        for field in fields
    }
    request = _request(tmp_path, [_record(0, {field: "benign" for field in fields})], tasks, 4)

    def fake_complete_json(system: str, _user: str, *, temperature: float = 0) -> dict:
        if "Task ID: failed" in system or '"description": "failed"' in system:
            return {"invalid": True}
        if system.startswith("Review one"):
            return {"approved": True, "reason": "Faithful."}
        return {"question": "Which status best describes this image?", "answer": "benign"}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    items = _read_items(result.raw_output_path)

    assert result.item_count == 3
    assert {item["task_id"] for item in items} == {"a", "b", "c"}


def test_bbox_review_remains_deterministic() -> None:
    annotation = {"finding": {"bbox": [1, 2, 8, 9]}}
    candidate = generation._Candidate(
        "detection",
        0,
        {"image_id": "img", "image_path": "img.dicom", "annotation": annotation},
        generation.project_supporting_annotation(annotation, ["annotation.finding.bbox[*]"]),
        _task(
            "annotation.finding.bbox[*]",
            task_type="open_vqa",
            response_format="bbox",
            options=None,
        ),
    )
    approved, _reason = generation._review(
        candidate,
        {
            "question": "Locate the finding.",
            "answer": [1.0, 2.0, 8.0, 9.0],
            "source_refs": [{"field": "annotation.finding.bbox[*]", "instance_index": None}],
        },
        {"language": "en"},
    )
    assert approved


def test_free_text_generation_retains_annotation_reference_validation(tmp_path: Path, monkeypatch) -> None:
    task = _task(
        "annotation.report",
        task_type="open_vqa",
        response_format="free_text",
        options=None,
    )
    request = _request(tmp_path, [_record(0, {"report": "Normal study."})], {"report": task}, 1)

    def fake_complete_json(system: str, _user: str, *, temperature: float = 0) -> dict:
        if system.startswith("Independently"):
            return {"approved": True, "reason": "Faithful."}
        return {
            "question": "Generate the report.",
            "answer": "Normal study.",
            "source_refs": [{"field": "annotation.report", "instance_index": None}],
        }

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    item = _read_items(generation.generate_raw_benchmark(request).raw_output_path)[0]

    assert item["options"] is None
    assert item["answer"] == "Normal study."
