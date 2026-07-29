import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import medbenchagent.generation.engine as generation
import medbenchagent.generation.free_text as free_text
import medbenchagent.generation.grounding as grounding
import medbenchagent.generation.vqa as vqa
from medbenchagent.dataset import prepare_annotation_dataset
from medbenchagent.generation.context import evidence_provenance, validate_evidence


def _record(index: int, annotation: dict) -> dict:
    return {
        "patient": {"patient_id": None, "annotation": {}},
        "case": {"case_id": None, "annotation": {}},
        "study": {"study_id": None, "annotation": {}},
        "image": {
            "image_id": f"img-{index}",
            "image_path": f"images/{index}.dicom",
            "modality": "Mammography",
            "anatomy": "Breast",
            "annotation": annotation,
        },
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
        "generation_logic": "Use only the approved supporting fields.",
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
        "run_id": "00000000-0000-0000-0000-000000000000",
        "dataset_id": "test-dataset",
        "blueprint_sha256": "a" * 64,
        "construction_events_path": str(tmp_path / "run" / "construction_events.jsonl"),
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
        field_line = next(
            line for line in system.splitlines() if line.startswith("- Supporting fields: ")
        )
        data_line = next(
            line for line in system.splitlines() if line.startswith("- Supporting data: ")
        )
        path = json.loads(field_line.removeprefix("- Supporting fields: "))[0]
        supporting_data = json.loads(data_line.removeprefix("- Supporting data: "))
        value = supporting_data
        for segment in path.split("."):
            value = value[segment]
        return {
            "question": "Which status best describes this image?",
            "answer": "benign",
            "evidence": {"path": path, "indices": [], "value": value},
        }

    return complete_json


def _candidate(task_id: str, annotation: dict, supporting_data: dict, task: dict):
    modality = annotation.pop("modality", None)
    record = _record(0, annotation)
    record["image"]["modality"] = modality
    return SimpleNamespace(
        task_id=task_id,
        record=record,
        supporting_record={"image": {"annotation": supporting_data}},
        task=task,
    )


def test_generation_prompts_use_image_modality_and_supporting_data() -> None:
    task = _task("letter", ["image.annotation.status"])
    candidate = _candidate(
        "vqa",
        {"modality": "Mammography"},
        {"status": "benign"},
        task,
    )

    prompt = vqa._prompt(candidate, {"language": "en", "modality": "CT"})

    assert "- Modality: Mammography" in prompt
    assert "- Supporting data:" in prompt
    assert "- Supporting annotation:" not in prompt
    missing_modality = _candidate("vqa", {}, {"status": "benign"}, task)
    assert "- Modality: " in vqa._prompt(missing_modality, {"language": "en", "modality": "CT"})
    assert "- Modality: CT" not in vqa._prompt(missing_modality, {"language": "en", "modality": "CT"})


def test_vqa_evidence_requires_exact_scalar_source_location() -> None:
    annotation = {
        "findings": [
            {"categories": ["mass", "asymmetry"]},
            {"categories": ["mass"]},
        ]
    }
    fields = ["image.annotation.findings[*].categories[*]"]
    valid = {
        "path": fields[0],
        "indices": [1, 0],
        "value": "mass",
    }

    record = _record(0, annotation)
    assert validate_evidence(record, fields, valid) == valid
    provenance = evidence_provenance(valid)
    assert provenance == {
        "path": fields[0],
        "indices": [1, 0],
        "evidence_sha256": provenance["evidence_sha256"],
    }
    assert "value" not in provenance

    invalid = [
        valid | {"path": "image.annotation.other"},
        valid | {"indices": [1]},
        valid | {"indices": [2, 0]},
        valid | {"indices": [True, 0]},
        valid | {"value": "Mass"},
    ]
    for evidence in invalid:
        with pytest.raises(ValueError):
            validate_evidence(record, fields, evidence)

    with pytest.raises(ValueError, match="scalar"):
        validate_evidence(
            _record(0, {"findings": [{"categories": [["mass"]]}]}),
            fields,
            valid | {"indices": [0, 0], "value": ["mass"]},
        )


def test_vqa_repairs_invalid_evidence_before_review(monkeypatch) -> None:
    task = _task("letter", ["image.annotation.status"])
    candidate = _candidate("vqa", {"status": "BENIGN"}, {"status": "BENIGN"}, task)
    responses = iter(
        [
            {
                "question": "Which status?",
                "answer": "benign",
                "evidence": {
                    "path": "image.annotation.status",
                    "indices": [],
                    "value": "Benign",
                },
            },
            {
                "question": "Which status?",
                "answer": "benign",
                "evidence": {
                    "path": "image.annotation.status",
                    "indices": [],
                    "value": "BENIGN",
                },
            },
        ]
    )
    monkeypatch.setattr(vqa, "complete_json", lambda *_args, **_kwargs: next(responses))

    generated = vqa.generate(candidate, {"language": "en"}, 0.7)

    assert generated["evidence"]["value"] == "BENIGN"


def test_grounding_modes_require_different_annotation_context() -> None:
    detection = _candidate(
        "detection",
        {"modality": "Mammography"},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["image.annotation.bbox[*]"]),
    )
    visual = _candidate(
        "visual_grounding",
        {"modality": "Mammography"},
        {"findings": [{"finding_categories": ["Mass"], "bbox": [1, 2, 8, 9]}]},
        _task(
            "bbox",
            ["image.annotation.findings[*].finding_categories[*]", "image.annotation.findings[*].bbox[*]"],
        ),
    )
    unnamed_visual = _candidate(
        "visual_grounding",
        {"modality": "Mammography"},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["image.annotation.bbox[*]"]),
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
            "vqa": _task("letter", ["image.annotation.status"], sample_count=1),
            "grounding": _task("bbox", ["image.annotation.bbox[*]"], sample_count=1),
            "report": _task("free_text", ["image.annotation.report"], sample_count=1),
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
    provenance = _read_items(str(Path(request["output_directory"]) / "item_provenance.jsonl"))

    generation_prompts = [prompt for prompt, _temperature in calls if prompt.startswith("Generate one")]
    assert [prompt.splitlines()[0] for prompt in generation_prompts] == [
        "Generate one letter-choice question for a medical image VQA benchmark.",
        "Generate one grounding question for a medical image benchmark.",
        "Generate one free-text medical image benchmark item.",
    ]
    assert [item["task_id"] for item in items] == ["vqa", "grounding", "report"]
    assert [entry["item_id"] for entry in provenance] == [item["id"] for item in items]
    assert all(entry["source_row"] == 1 for entry in provenance)
    assert items[0]["answer"] == "A"
    assert items[1]["answer"] == [1.0, 2.0, 8.0, 9.0]
    assert items[2]["answer"] == "Normal study."
    assert all(set(item) == {"id", "task_id", "dataset", "image_id", "image_path", "question", "options", "answer"} for item in items)
    assert not list(Path(request["output_directory"]).glob("benchmark_items.*.progress.jsonl"))
    assert all(temperature == 0.7 for prompt, temperature in calls if prompt.startswith("Generate one"))
    assert all(temperature == 0 for prompt, temperature in calls if "Review one" in prompt)


def test_each_stage_applies_its_own_three_task_image_limit(tmp_path: Path, monkeypatch) -> None:
    annotation = {
        **{f"status_{index}": "benign" for index in range(4)},
        **{f"bbox_{index}": [1, 2, 8, 9] for index in range(4)},
    }
    tasks = {
        **{
            f"vqa_{index}": _task("letter", [f"image.annotation.status_{index}"], sample_count=1)
            for index in range(4)
        },
        **{
            f"grounding_{index}": _task("bbox", [f"image.annotation.bbox_{index}[*]"], sample_count=1)
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
    task = _task("bbox", ["image.annotation.bbox[*]"], sample_count=1)
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
        {"grounding": _task("bbox", ["image.annotation.boxes[*]"], sample_count=1)},
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
            "vqa": _task("letter", ["image.annotation.status"], sample_count=1),
            "grounding": _task("bbox", ["image.annotation.bbox[*]"], sample_count=1),
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
    task = _task("free_text", ["image.annotation.report"], sample_count=1)
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
    task = _task("letter", ["image.annotation.status", "image.annotation.laterality"], options=["A. benign", "B) malignant"])
    request = {
        "dataset_id": "test-dataset",
        "tasks": {"vqa": task},
        "target_sample_count": 1,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
    }
    generation._validate_tasks(request)
    assert task["options"] == ["benign", "malignant"]
    assert not generation._all_supporting_fields_present(
        _record(0, {"status": "benign", "laterality": ""}), task["supporting_fields"]
    )
    assert generation._all_supporting_fields_present(
        _record(0, {"status": "benign", "laterality": False}), task["supporting_fields"]
    )


def test_stage_progress_is_resumed_and_cleaned_only_after_raw_merge(tmp_path: Path) -> None:
    task = _task("letter", ["image.annotation.status"], sample_count=1)
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
