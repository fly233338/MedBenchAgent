import json
from pathlib import Path

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
        "medical_task_description": "This assesses recognition; references support terminology.",
        "citations": [],
        **extra,
    }


def _request(tmp_path: Path, records: list[dict], tasks: dict, target: int, seed: int = 0) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        "seed": seed,
        "balance_answer_positions": True,
        "output_directory": str(tmp_path / "run"),
    }


def _item(task_id: str, answer, field: str, question: str = "What is shown?", index=None) -> dict:
    return {
        "task_id": task_id,
        "question": question,
        "answer": answer,
        "source_refs": [{"field": field, "instance_index": index}],
    }


def _read_items(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_generation_routes_records_and_normalizes_letter_answer(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"view": "craniocaudal"}), _record(1, {"view": "MLO"})],
        {"view": _task("letter", ["annotation.view"], options=["CC", "MLO"])},
        1,
    )

    def fake_complete_json(system: str, prompt: str) -> dict:
        if system.startswith("Independently"):
            return {"approved": True, "reason": "Faithful."}
        if '"view": "MLO"' in prompt:
            return {"items": [_item("view", "MLO", "annotation.view")]}
        return {"items": []}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    item = _read_items(result.raw_output_path)[0]

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


def test_one_record_can_generate_multiple_reviewed_task_items(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"finding": {"kind": "mass", "bbox": [1, 2, 8, 9]}, "report": "Expert report."})],
        {
            "detection": _task(
                "bbox", ["annotation.finding.kind", "annotation.finding.bbox[*]"], sample_count=1
            ),
            "report": _task("free_text", ["annotation.report"], sample_count=1),
        },
        2,
    )

    def fake_complete_json(system: str, _prompt: str) -> dict:
        if system.startswith("Independently"):
            return {"approved": True, "reason": "Faithful."}
        return {
            "items": [
                _item("detection", [1, 2, 8, 9], "annotation.finding.bbox[*]", "Locate the mass."),
                _item("report", "Expert report.", "annotation.report", "Generate a report."),
            ]
        }

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    items = _read_items(result.raw_output_path)

    assert result.task_counts == {"detection": 1, "report": 1}
    assert items[0]["answer"] == [1.0, 2.0, 8.0, 9.0]
    assert items[1]["answer"] == "Expert report."


def test_generation_repairs_one_invalid_routed_response(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign"})],
        {"status": _task("letter", ["annotation.status"], options=["benign", "malignant"])},
        1,
    )
    calls = []

    def fake_complete_json(system: str, _prompt: str) -> dict:
        calls.append(system)
        if system.startswith("Select medically"):
            return {}
        if system.startswith("Repair one"):
            return {"items": [_item("status", "benign", "annotation.status")]}
        return {"approved": True, "reason": "Faithful."}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)

    assert result.item_count == 1
    assert sum(system.startswith("Repair one") for system in calls) == 1


def test_reviewer_rejection_finishes_with_empty_best_effort_output(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign"})],
        {"status": _task("letter", ["annotation.status"], options=["benign", "malignant"])},
        1,
    )
    monkeypatch.setattr(
        generation,
        "complete_json",
        lambda system, _prompt: {"approved": False, "reason": "Leaky."}
        if system.startswith("Independently")
        else {"items": [_item("status", "benign", "annotation.status")]},
    )

    result = generation.generate_raw_benchmark(request)

    assert result.item_count == 0
    assert Path(result.raw_output_path).read_text(encoding="utf-8") == ""


def test_dataset_exhaustion_returns_partial_output(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign"})],
        {"status": _task("letter", ["annotation.status"], options=["benign", "malignant"])},
        2,
    )
    monkeypatch.setattr(
        generation,
        "complete_json",
        lambda system, _prompt: {"approved": True, "reason": "Faithful."}
        if system.startswith("Independently")
        else {"items": [_item("status", "benign", "annotation.status")]},
    )

    result = generation.generate_raw_benchmark(request)

    assert result.item_count == 1
    assert result.task_counts == {"status": 1}


def test_seeded_record_order_is_reproducible_and_changes_with_seed(tmp_path: Path) -> None:
    records = [_record(index, {"status": "benign"}) for index in range(5)]
    task = _task("letter", ["annotation.status"], options=["benign", "malignant"])
    request_zero = _request(tmp_path / "zero", records, {"status": task}, 1, seed=0)
    request_one = _request(tmp_path / "one", records, {"status": task}, 1, seed=1)
    tasks_zero, targets_zero = generation._validate_tasks(request_zero)
    tasks_one, targets_one = generation._validate_tasks(request_one)

    _, first_zero = generation._next_candidate_record(
        request_zero, tasks_zero, targets_zero, {"status": 0}, 0
    )
    _, repeated_zero = generation._next_candidate_record(
        request_zero, tasks_zero, targets_zero, {"status": 0}, 0
    )
    _, first_one = generation._next_candidate_record(
        request_one, tasks_one, targets_one, {"status": 0}, 0
    )

    assert first_zero[0].source_row == repeated_zero[0].source_row
    assert first_zero[0].source_row != first_one[0].source_row


def test_progress_merge_is_idempotent(tmp_path: Path) -> None:
    candidate = generation._Candidate(
        "status",
        0,
        {
            "image_id": "img-0",
            "image_path": "images/0.png",
            "annotation": {"status": "benign"},
        },
        {"status": "benign"},
        _task("letter", ["annotation.status"], options=["benign", "malignant"]),
    )
    item = generation._build_item(
        candidate,
        {"question": "What is the status?", "answer": "benign", "source_refs": [{"field": "annotation.status", "instance_index": None}]},
    )
    path = tmp_path / "progress.jsonl"

    generation._merge_progress(path, [item], {"status": 0})
    merged = generation._merge_progress(path, [item], {"status": 0})

    assert len(merged) == 1
