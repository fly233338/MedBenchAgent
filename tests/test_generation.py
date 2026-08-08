import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import medbenchagent.generation.engine as generation
import medbenchagent.generation.free_text as free_text
import medbenchagent.generation.grounding as grounding
import medbenchagent.generation.vqa as vqa
from medbenchagent.dataset import prepare_annotation_dataset
from medbenchagent.generation.context import (
    evidence_provenance,
    locate_bbox_evidence,
    locate_scalar_evidence,
)


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
        "metric": {"letter": "accuracy", "free_text": "bleu", "bbox": "miou"}[response_format],
        "supporting_fields": fields,
        "dataset_support_reason": "Available.",
        "question_templates": ["What is shown in this image?"],
        "generation_logic": "Use the approved annotation and selected template to generate the question.",
        "medical_task_description": "Recognition task.",
        "citations": [],
    }
    if response_format == "letter":
        task["options"] = ["benign", "malignant"]
    task.update(extra)
    if response_format == "letter" and "answer_value_mapping" not in task:
        task["answer_field"] = task.get("answer_field", fields[0])
        normalized_options = [
            re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", option).strip()
            for option in task["options"]
        ]
        task["answer_value_mapping"] = [
            {"raw_value": option, "option": option} for option in normalized_options
        ]
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
            return {"question": "Locate the finding."}
        if system.startswith("Generate one free-text"):
            return {"question": "What is the report?"}
        if system.startswith("Generate only the question stem"):
            return {"question": "Which status best describes this image?"}
        raise AssertionError(f"Unexpected generation prompt: {system}")

    return complete_json


def _candidate(task_id: str, annotation: dict, supporting_data: dict, task: dict):
    modality = annotation.pop("modality", None)
    record = _record(0, annotation)
    record["image"]["modality"] = modality
    expected_answer = None
    expected_evidence = None
    if task.get("response_format") == "letter":
        expected_answer = task["options"][0]
        expected_evidence = {
            "path": task["answer_field"],
            "indices": [],
            "value": supporting_data[next(iter(supporting_data))],
        }
    return SimpleNamespace(
        task_id=task_id,
        record=record,
        supporting_record={"image": {"annotation": supporting_data}},
        task=task,
        question_template=task["question_templates"][0],
        expected_answer=expected_answer,
        expected_evidence=expected_evidence,
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
    assert "- Selected template: What is shown in this image?" in prompt
    assert "- Medical task description: Recognition task." in prompt
    assert "- Generation logic: Use the approved annotation and selected template to generate the question." in prompt
    assert "- Task definition:" not in prompt
    assert "- Fixed answer option: benign" in prompt
    assert "- Fixed raw evidence:" in prompt
    assert "- Supporting data:" not in prompt
    missing_modality = _candidate("vqa", {}, {"status": "benign"}, task)
    assert "- Modality: " in vqa._prompt(missing_modality, {"language": "en", "modality": "CT"})
    assert "- Modality: CT" not in vqa._prompt(missing_modality, {"language": "en", "modality": "CT"})


def test_mapped_vqa_review_prompt_uses_fixed_context_and_rules(monkeypatch) -> None:
    task = _task("letter", ["image.annotation.status"])
    candidate = _candidate("vqa", {}, {"status": "benign"}, task)
    captured: dict[str, str] = {}

    def complete(system: str, _user: str, **_kwargs) -> dict[str, object]:
        captured["prompt"] = system
        return {"approved": True, "reason": "Faithful."}

    monkeypatch.setattr(vqa, "complete_json", complete)
    approved, reason = vqa.review(
        candidate,
        {
            "question": "Which status is shown?",
            "answer": candidate.expected_answer,
            "evidence": candidate.expected_evidence,
        },
        {"language": "en"},
    )

    assert approved is True
    assert reason == "Faithful."
    prompt = captured["prompt"]
    assert "- Answer field: image.annotation.status" in prompt
    assert "- Answer mapping:" in prompt
    assert "- Selected question template:" in prompt
    assert "fixed answer and evidence were deterministically derived by code" in prompt
    assert "8. Ambiguity" in prompt
    assert "- Medical task description: Recognition task." in prompt
    assert "- Generation logic: Use the approved annotation and selected template to generate the question." in prompt
    assert "- Task definition:" not in prompt


def test_question_template_selection_is_seeded_and_supports_single_template() -> None:
    task = _task("letter", ["image.annotation.status"])
    assert generation._select_question_template(task, 7, "status", "img-1", 1) == task[
        "question_templates"
    ][0]

    task["question_templates"] = ["one", "two", "three"]
    first = generation._select_question_template(task, 7, "status", "img-1", 1)
    assert generation._select_question_template(task, 7, "status", "img-1", 1) == first
    assert len(
        {
            generation._select_question_template(task, 7, "status", f"img-{index}", index)
            for index in range(20)
        }
    ) > 1


def test_all_generation_formats_use_the_selected_question_template() -> None:
    selected = "Use this selected anchor."
    vqa_candidate = _candidate(
        "vqa", {"status": "benign"}, {"status": "benign"}, _task("letter", ["image.annotation.status"])
    )
    grounding_candidate = _candidate(
        "detection",
        {"bbox": [1, 2, 8, 9]},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["image.annotation.bbox[*]"]),
    )
    free_text_candidate = _candidate(
        "report",
        {"report": "Normal study."},
        {"report": "Normal study."},
        _task("free_text", ["image.annotation.report"]),
    )
    for candidate in (vqa_candidate, grounding_candidate, free_text_candidate):
        candidate.question_template = selected

    prompts = [
        vqa._prompt(vqa_candidate, {"language": "en"}),
        grounding._prompt(grounding_candidate, {"language": "en"}),
        free_text._prompt(free_text_candidate, {"language": "en"}),
    ]
    for prompt in prompts:
        assert selected in prompt
        assert "- Medical task description: Recognition task." in prompt
        assert "- Generation logic: Use the approved annotation and selected template to generate the question." in prompt
        assert "- Task definition:" not in prompt


def test_bbox_and_free_text_review_prompts_use_design_fields(monkeypatch) -> None:
    grounding_candidate = _candidate(
        "detection",
        {"bbox": [1, 2, 8, 9]},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["image.annotation.bbox[*]"]),
    )
    grounding_prompt: dict[str, str] = {}

    def capture_grounding(system: str, _user: str, **_kwargs) -> dict[str, object]:
        grounding_prompt["value"] = system
        return {"approved": True, "reason": "Faithful."}

    monkeypatch.setattr(grounding, "complete_json", capture_grounding)
    grounding.review(
        grounding_candidate,
        {"question": "Where is the lesion?", "answer": [1.0, 2.0, 8.0, 9.0]},
        {"language": "en"},
    )

    free_text_candidate = _candidate(
        "report",
        {"report": "Normal study."},
        {"report": "Normal study."},
        _task("free_text", ["image.annotation.report"]),
    )
    free_text_prompt: dict[str, str] = {}

    def capture_free_text(system: str, _user: str, **_kwargs) -> dict[str, object]:
        free_text_prompt["value"] = system
        return {"approved": True, "reason": "Faithful."}

    monkeypatch.setattr(free_text, "complete_json", capture_free_text)
    free_text.review(
        free_text_candidate,
        {"question": "What is the report?", "answer": "Normal study."},
        {"language": "en"},
    )

    for prompt in (grounding_prompt["value"], free_text_prompt["value"]):
        assert "- Medical task description: Recognition task." in prompt
        assert "- Generation logic: Use the approved annotation and selected template to generate the question." in prompt
        assert "- Task definition:" not in prompt


def test_direct_answer_evidence_requires_one_exact_source_location() -> None:
    record = _record(
        0,
        {
            "report": "Normal study.",
            "findings": [
                {"label": "mass", "bbox": [1, 2, 8, 9]},
                {"label": "calcification", "bbox": [10, 11, 20, 21]},
            ],
        },
    )
    scalar = locate_scalar_evidence(
        record,
        ["image.annotation.report"],
        "Normal study.",
    )
    assert scalar == {
        "path": "image.annotation.report",
        "indices": [],
        "value": "Normal study.",
    }
    bbox = locate_bbox_evidence(
        record,
        ["image.annotation.findings[*].bbox[*]"],
        [10.0, 11.0, 20.0, 21.0],
    )
    assert [coordinate["indices"] for coordinate in bbox] == [
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
    ]
    compact = evidence_provenance(bbox)
    assert isinstance(compact, list)
    assert all("value" not in coordinate for coordinate in compact)

    with pytest.raises(ValueError, match="exactly one"):
        locate_scalar_evidence(
            _record(0, {"first": "same", "second": "same"}),
            ["image.annotation.first", "image.annotation.second"],
            "same",
        )
    with pytest.raises(ValueError, match="valid xyxy"):
        locate_bbox_evidence(
            record,
            ["image.annotation.findings[*].bbox[*]"],
            [1, 2, float("inf"), 9],
        )


def test_mapped_vqa_repairs_invalid_response_contract(monkeypatch) -> None:
    task = _task("letter", ["image.annotation.status"])
    candidate = _candidate("vqa", {"status": "BENIGN"}, {"status": "BENIGN"}, task)
    responses = iter(
        [
            {"question": "Which status?", "answer": "benign"},
            {"question": "Which status?"},
        ]
    )
    monkeypatch.setattr(vqa, "complete_json", lambda *_args, **_kwargs: next(responses))

    generated = vqa.generate(candidate, {"language": "en"}, 0.7)

    assert generated["answer"] == candidate.expected_answer
    assert generated["evidence"]["value"] == "BENIGN"


def test_grounding_modes_require_different_annotation_context() -> None:
    detection = _candidate(
        "detection",
        {"modality": "Mammography", "bbox": [1, 2, 8, 9]},
        {"bbox": [1, 2, 8, 9]},
        _task("bbox", ["image.annotation.bbox[*]"]),
    )
    visual = _candidate(
        "visual_grounding",
        {
            "modality": "Mammography",
            "findings": [{"finding_categories": ["Mass"], "bbox": [1, 2, 8, 9]}],
        },
        {"findings": [{"finding_categories": ["Mass"], "bbox": [1, 2, 8, 9]}]},
        _task(
            "bbox",
            ["image.annotation.findings[*].finding_categories[*]", "image.annotation.findings[*].bbox[*]"],
        ),
    )
    unnamed_visual = _candidate(
        "visual_grounding",
        {"modality": "Mammography", "bbox": [1, 2, 8, 9]},
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

    generation_prompts = [prompt for prompt, _temperature in calls if prompt.startswith("Generate")]
    assert [prompt.splitlines()[0] for prompt in generation_prompts] == [
        "Generate only the question stem for one letter-choice medical-image benchmark item.",
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
    assert isinstance(provenance[0]["answer_evidence"], dict)
    assert [coordinate["indices"] for coordinate in provenance[1]["answer_evidence"]] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert isinstance(provenance[2]["answer_evidence"], dict)
    assert all(
        "value" not in witness
        for entry in provenance
        for witness in (
            entry["answer_evidence"]
            if isinstance(entry["answer_evidence"], list)
            else [entry["answer_evidence"]]
        )
    )
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
            {"question": "Locate the finding."},
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
            {"question": "Locate the finding."},
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
            {"question": "What is the report?"},
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
    task = _task(
        "letter",
        ["image.annotation.status", "image.annotation.laterality"],
        options=["A. benign", "B) malignant"],
        sample_count=1,
    )
    request = {
        "dataset_id": "test-dataset",
        "tasks": {"vqa": task},
        "target_sample_count": 1,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
    }
    generation._validate_tasks(request)
    assert task["options"] == ["benign", "malignant"]
    task["question_templates"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="unique non-empty"):
        generation._validate_tasks(request)
    assert not generation._all_supporting_fields_present(
        _record(0, {"status": "benign", "laterality": ""}), task["supporting_fields"]
    )
    assert generation._all_supporting_fields_present(
        _record(0, {"status": "benign", "laterality": False}), task["supporting_fields"]
    )


def test_letter_tasks_require_structured_answer_mapping() -> None:
    task = _task("letter", ["image.annotation.status"], sample_count=1)
    task.pop("answer_field")
    task.pop("answer_value_mapping")
    request = {
        "dataset_id": "test-dataset",
        "tasks": {"vqa": task},
        "target_sample_count": 1,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
    }

    with pytest.raises(ValueError, match="require answer_field and answer_value_mapping"):
        generation._validate_tasks(request)


def test_generation_rejects_final_task_without_metric(tmp_path: Path) -> None:
    task = _task("letter", ["image.annotation.status"], sample_count=1)
    task.pop("metric")
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"task": task}, 1)

    with pytest.raises(ValueError, match="metric"):
        generation.initialize_generation(request)


def test_generation_rejects_final_task_without_sample_count(tmp_path: Path) -> None:
    task = _task("letter", ["image.annotation.status"])
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"task": task}, 1)

    with pytest.raises(ValueError, match="sample_count"):
        generation.initialize_generation(request)


def test_structured_mapping_fixes_answer_and_evidence_before_vqa_generation(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(
        "letter",
        ["image.annotation.status"],
        sample_count=1,
        parent_task_id="finding_attribute",
        question_type_id="status",
        answer_field="image.annotation.status",
        answer_value_mapping=[
            {"raw_value": "benign", "option": "benign"},
            {"raw_value": "malignant", "option": "malignant"},
        ],
    )
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"status": task}, 1)
    calls: list[str] = []
    responses = iter(
        [
            {"question": "Which status is shown?"},
            {"approved": True, "reason": "Faithful."},
        ]
    )

    def complete(system: str, *_args, **_kwargs) -> dict:
        calls.append(system)
        return next(responses)

    monkeypatch.setattr(vqa, "complete_json", complete)
    result = generation.generate_raw_benchmark(request)
    item = _read_items(result.raw_output_path)[0]
    provenance = _read_items(result.provenance_output_path)[0]

    assert item["answer"] == "A"
    assert "Fixed answer option: benign" in calls[0]
    assert "Supporting data:" not in calls[0]
    assert provenance["answer_evidence"]["path"] == "image.annotation.status"


def test_structured_mapping_skips_unmapped_or_multiple_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    mapping = [
        {"raw_value": "low", "option": "benign"},
        {"raw_value": "high", "option": "malignant"},
    ]
    unmapped_task = _task(
        "letter",
        ["image.annotation.signal"],
        sample_count=1,
        answer_field="image.annotation.signal",
        answer_value_mapping=mapping,
    )
    unmapped = _request(
        tmp_path / "unmapped",
        [_record(0, {"signal": "intermediate"})],
        {"signal": unmapped_task},
        1,
    )
    monkeypatch.setattr(vqa, "complete_json", lambda *_args, **_kwargs: pytest.fail("model called"))
    assert generation.generate_raw_benchmark(unmapped).item_count == 0

    multiple_task = _task(
        "letter",
        ["image.annotation.findings[*].signal"],
        sample_count=1,
        answer_field="image.annotation.findings[*].signal",
        answer_value_mapping=mapping,
    )
    multiple = _request(
        tmp_path / "multiple",
        [_record(0, {"findings": [{"signal": "low"}, {"signal": "high"}]})],
        {"signal": multiple_task},
        1,
    )
    assert generation.generate_raw_benchmark(multiple).item_count == 0


def test_stage_progress_is_resumed_and_cleaned_only_after_raw_merge(tmp_path: Path) -> None:
    task = _task("letter", ["image.annotation.status"], sample_count=1)
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"vqa": task}, 1)
    output_dir = Path(request["output_directory"])
    output_dir.mkdir(parents=True)
    progress = output_dir / "benchmark_items.vqa.progress.jsonl"
    progress.write_text(
        json.dumps(
            {
                "item": {
                    "id": "existing",
                    "task_id": "vqa",
                    "dataset": "test-dataset",
                    "image_id": "img-0",
                    "image_path": "images/0.dicom",
                    "question": "Which status?",
                    "options": ["benign", "malignant"],
                    "answer": "A",
                },
                "provenance": {
                    "item_id": "existing",
                    "answer_evidence": {
                        "path": "image.annotation.status",
                        "indices": [],
                        "evidence_sha256": "a" * 64,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    state = generation.initialize_generation(request)
    assert generation.generation_done(state)
    result = generation.finalize_generation(request)

    assert _read_items(result.raw_output_path)[0]["id"] == "existing"
    assert _read_items(result.provenance_output_path)[0]["item_id"] == "existing"
    assert not progress.exists()

def test_vqa_prompt_requires_distinct_question_structure() -> None:
    assert "distinctly different grammatical structure" in vqa.VQA_GENERATION_PROMPT
    assert "Based on…" in vqa.VQA_GENERATION_PROMPT
    assert "On this mammogram" not in vqa.VQA_GENERATION_PROMPT
