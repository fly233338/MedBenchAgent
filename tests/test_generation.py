import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import medbenchagent.generation.engine as generation
import medbenchagent.generation.free_text as free_text
import medbenchagent.generation.grounding as grounding
import medbenchagent.generation.vqa as vqa
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


def _task(response_format: str, fields: list[str], **extra) -> dict:
    task = {
        "task_type": "close_vqa" if response_format == "letter" else "open_vqa",
        "response_format": response_format,
        "description": f"{response_format} task",
        "require": "Use approved annotation",
        "metric": "accuracy",
        "supporting_fields": fields,
        "dataset_support_reason": "Available.",
        "question_templates": ["What is shown in this image?"],
        "generation_logic": "Use only the approved annotation.",
        "medical_task_description": "Recognition task.",
        "citations": [],
    }
    if response_format == "letter":
        task["options"] = ["benign", "malignant"]
    task.update(extra)
    return task


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


def _fake_model(calls: list[tuple[str, float]]):
    def complete_json(system: str, _user: str, *, temperature: float = 0) -> dict:
        calls.append((system, temperature))
        if "Review one" in system:
            return {"approved": True, "reason": "Faithful."}
        if system.startswith("Generate one grounding"):
            return {"question": "Locate the finding.", "answer": [1, 2, 8, 9]}
        if system.startswith("Generate one free-text"):
            return {"question": "What is the report?", "answer": "Normal study."}
        return {"question": "Which status best describes this image?", "answer": "benign"}

    return complete_json


def _candidate(task_id: str, annotation: dict, supporting_annotation: dict, task: dict):
    return SimpleNamespace(
        task_id=task_id,
        record={"annotation": annotation},
        supporting_annotation=supporting_annotation,
        task=task,
    )


def test_generation_prompts_use_annotation_modality() -> None:
    task = _task("letter", ["annotation.status"])
    candidate = _candidate(
        "vqa",
        {"modality": "Mammography"},
        {"status": "benign"},
        task,
    )

    prompt = vqa._prompt(candidate, {"language": "en", "modality": "CT"})

    assert "- Modality: Mammography" in prompt
    missing_modality = _candidate("vqa", {}, {"status": "benign"}, task)
    assert "- Modality: " in vqa._prompt(missing_modality, {"language": "en", "modality": "CT"})
    assert "- Modality: CT" not in vqa._prompt(missing_modality, {"language": "en", "modality": "CT"})


def test_grounding_modes_require_different_annotation_context() -> None:
    detection = _candidate(
        "detection",
        {"modality": "Mammography"},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["annotation.bbox[*]"]),
    )
    visual = _candidate(
        "visual_grounding",
        {"modality": "Mammography"},
        {"findings": [{"finding_categories": ["Mass"], "bbox": [1, 2, 8, 9]}]},
        _task(
            "bbox",
            ["annotation.findings[*].finding_categories[*]", "annotation.findings[*].bbox[*]"],
        ),
    )
    unnamed_visual = _candidate(
        "visual_grounding",
        {"modality": "Mammography"},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["annotation.bbox[*]"]),
    )

    assert grounding.supports(detection)
    assert grounding.supports(visual)
    assert not grounding.supports(unnamed_visual)
    assert "Do not name, describe" in grounding._prompt(detection, {"language": "en"})
    assert "explicitly named finding 'Mass'" in grounding._prompt(visual, {"language": "en"})


def test_three_stages_run_in_order_and_write_compact_raw(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"status": "benign", "bbox": [1, 2, 8, 9], "report": "Normal study."})],
        {
            "vqa": _task("letter", ["annotation.status"], sample_count=1),
            "grounding": _task("bbox", ["annotation.bbox[*]"], sample_count=1),
            "report": _task("free_text", ["annotation.report"], sample_count=1),
        },
        3,
    )
    calls: list[tuple[str, float]] = []
    fake = _fake_model(calls)
    monkeypatch.setattr(vqa, "complete_json", fake)
    monkeypatch.setattr(grounding, "complete_json", fake)
    monkeypatch.setattr(free_text, "complete_json", fake)

    result = generation.generate_raw_benchmark(request)
    items = _read_items(result.raw_output_path)

    generation_prompts = [prompt for prompt, _temperature in calls if prompt.startswith("Generate one")]
    assert [prompt.splitlines()[0] for prompt in generation_prompts] == [
        "Generate one letter-choice question for a medical image VQA benchmark.",
        "Generate one grounding question for a medical image benchmark.",
        "Generate one free-text medical image benchmark item.",
    ]
    assert [item["task_id"] for item in items] == ["vqa", "grounding", "report"]
    assert items[0]["answer"] == "A"
    assert items[1]["answer"] == [1.0, 2.0, 8.0, 9.0]
    assert items[2]["answer"] == "Normal study."
    assert all(set(item) == {"id", "task_id", "image_id", "image_path", "question", "options", "answer"} for item in items)
    assert not list(Path(request["output_directory"]).glob("benchmark_items.*.progress.jsonl"))
    assert all(temperature == 0.7 for prompt, temperature in calls if prompt.startswith("Generate one"))
    assert all(temperature == 0 for prompt, temperature in calls if "Review one" in prompt)


def test_stages_restart_database_cursor_and_keep_independent_limits(tmp_path: Path, monkeypatch) -> None:
    records = [_record(index, {"status": "benign", "bbox": [1, 2, 8, 9]}) for index in range(2)]
    request = _request(
        tmp_path,
        records,
        {
            "vqa": _task("letter", ["annotation.status"], sample_count=1),
            "grounding": _task("bbox", ["annotation.bbox[*]"], sample_count=1),
        },
        2,
    )
    calls: list[tuple[str, float]] = []
    fake = _fake_model(calls)
    monkeypatch.setattr(vqa, "complete_json", fake)
    monkeypatch.setattr(grounding, "complete_json", fake)
    cursors: list[tuple[str, int]] = []
    original = generation._next_candidate_record

    def wrapped(request, tasks, targets, completed, cursor, handler):
        cursors.append((handler.__name__, cursor))
        return original(request, tasks, targets, completed, cursor, handler)

    monkeypatch.setattr(generation, "_next_candidate_record", wrapped)
    generation.generate_raw_benchmark(request)

    assert ("medbenchagent.generation.vqa", 0) in cursors
    assert ("medbenchagent.generation.grounding", 0) in cursors


def test_each_stage_applies_its_own_three_task_image_limit(tmp_path: Path, monkeypatch) -> None:
    annotation = {
        **{f"status_{index}": "benign" for index in range(4)},
        **{f"bbox_{index}": [1, 2, 8, 9] for index in range(4)},
    }
    tasks = {
        **{
            f"vqa_{index}": _task("letter", [f"annotation.status_{index}"], sample_count=1)
            for index in range(4)
        },
        **{
            f"grounding_{index}": _task("bbox", [f"annotation.bbox_{index}[*]"], sample_count=1)
            for index in range(4)
        },
    }
    request = _request(tmp_path, [_record(0, annotation)], tasks, 8)
    calls: list[tuple[str, float]] = []
    fake = _fake_model(calls)
    monkeypatch.setattr(vqa, "complete_json", fake)
    monkeypatch.setattr(grounding, "complete_json", fake)

    result = generation.generate_raw_benchmark(request)
    items = _read_items(result.raw_output_path)

    assert len([item for item in items if item["task_id"].startswith("vqa_")]) == 3
    assert len([item for item in items if item["task_id"].startswith("grounding_")]) == 3
    assert result.item_count == 6


def test_grounding_requires_exactly_one_bbox_and_repairs_once(tmp_path: Path, monkeypatch) -> None:
    task = _task("bbox", ["annotation.bbox[*]"], sample_count=1)
    request = _request(tmp_path, [_record(0, {"bbox": [1, 2, 8, 9]})], {"grounding": task}, 1)
    responses = iter(
        [
            {"question": "Locate the finding.", "answer": [0, 0, 1, 1]},
            {"question": "Locate the finding.", "answer": [1, 2, 8, 9]},
            {"approved": True, "reason": "Faithful."},
        ]
    )
    temperatures: list[float] = []

    def fake(_system: str, _user: str, *, temperature: float = 0) -> dict:
        temperatures.append(temperature)
        return next(responses)

    monkeypatch.setattr(grounding, "complete_json", fake)
    item = _read_items(generation.generate_raw_benchmark(request).raw_output_path)[0]

    assert item["answer"] == [1.0, 2.0, 8.0, 9.0]
    assert temperatures == [0.7, 0.7, 0]

    multi = _request(
        tmp_path / "multi",
        [_record(0, {"boxes": [[1, 2, 8, 9], [10, 11, 20, 21]]})],
        {"grounding": _task("bbox", ["annotation.boxes[*]"], sample_count=1)},
        1,
    )
    monkeypatch.setattr(grounding, "complete_json", lambda *_args, **_kwargs: pytest.fail("model called"))
    result = generation.generate_raw_benchmark(multi)
    assert result.task_counts == {"grounding": 0}


def test_stage_shortfall_does_not_block_later_stage(tmp_path: Path, monkeypatch) -> None:
    request = _request(
        tmp_path,
        [_record(0, {"bbox": [1, 2, 8, 9]})],
        {
            "vqa": _task("letter", ["annotation.status"], sample_count=1),
            "grounding": _task("bbox", ["annotation.bbox[*]"], sample_count=1),
        },
        2,
    )
    monkeypatch.setattr(vqa, "complete_json", lambda *_args, **_kwargs: pytest.fail("VQA model called"))
    responses = iter(
        [
            {"question": "Locate the finding.", "answer": [1, 2, 8, 9]},
            {"approved": True, "reason": "Faithful."},
        ]
    )
    monkeypatch.setattr(grounding, "complete_json", lambda *_args, **_kwargs: next(responses))

    result = generation.generate_raw_benchmark(request)

    assert result.task_counts == {"vqa": 0, "grounding": 1}


def test_free_text_requires_exact_unique_reference_text_and_repairs_once(tmp_path: Path, monkeypatch) -> None:
    task = _task("free_text", ["annotation.report"], sample_count=1)
    request = _request(tmp_path, [_record(0, {"report": "Normal study."})], {"report": task}, 1)
    responses = iter(
        [
            {"question": "What is the report?", "answer": "Normal."},
            {"question": "What is the report?", "answer": "Normal study."},
            {"approved": True, "reason": "Faithful."},
        ]
    )
    monkeypatch.setattr(free_text, "complete_json", lambda *_args, **_kwargs: next(responses))
    item = _read_items(generation.generate_raw_benchmark(request).raw_output_path)[0]
    assert item["answer"] == "Normal study."

    missing = _request(
        tmp_path / "missing",
        [_record(0, {"report": ""})],
        {"report": task},
        1,
    )
    monkeypatch.setattr(free_text, "complete_json", lambda *_args, **_kwargs: pytest.fail("model called"))
    assert generation.generate_raw_benchmark(missing).item_count == 0


def test_all_supporting_fields_must_be_nonempty_and_vqa_options_remain_normalized() -> None:
    task = _task("letter", ["annotation.status", "annotation.laterality"], options=["A. benign", "B) malignant"])
    request = {
        "tasks": {"vqa": task},
        "target_sample_count": 1,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
    }
    generation._validate_tasks(request)
    assert task["options"] == ["benign", "malignant"]
    assert not generation._all_supporting_fields_present({"status": "benign", "laterality": ""}, task["supporting_fields"])
    assert generation._all_supporting_fields_present({"status": "benign", "laterality": False}, task["supporting_fields"])


def test_stage_progress_is_resumed_and_cleaned_only_after_raw_merge(tmp_path: Path) -> None:
    task = _task("letter", ["annotation.status"], sample_count=1)
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"vqa": task}, 1)
    output_dir = Path(request["output_directory"])
    output_dir.mkdir(parents=True)
    progress = output_dir / "benchmark_items.vqa.progress.jsonl"
    progress.write_text(
        json.dumps(
            {
                "id": "existing",
                "task_id": "vqa",
                "image_id": "img-0",
                "image_path": "images/0.dicom",
                "question": "Which status?",
                "options": ["benign", "malignant"],
                "answer": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = generation.initialize_generation(request)
    assert generation.generation_done(state)
    result = generation.finalize_generation(request, state)

    assert _read_items(result.raw_output_path)[0]["id"] == "existing"
    assert not progress.exists()

def test_vqa_prompt_requires_distinct_question_structure() -> None:
    assert "distinctly different grammatical structure" in vqa.VQA_GENERATION_PROMPT
    assert "Based on…" in vqa.VQA_GENERATION_PROMPT
    assert "On this mammogram" not in vqa.VQA_GENERATION_PROMPT
