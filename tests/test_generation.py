import json
import re
from pathlib import Path
from types import SimpleNamespace
import pytest
import medbenchagent.generation.engine as generation
import medbenchagent.generation.grounding as grounding
from medbenchagent.answer_mapping import canonical_answer_scalar
from medbenchagent.data.dataset import prepare_annotation_dataset
from medbenchagent.generation.context import locate_bbox_evidence, locate_scalar_evidence
from medbenchagent.generation.templates import validate_templates
from medbenchagent.planning.protocol import TemplateSpecification
from medbenchagent.settings import get_settings


def test_template_slots_exclude_answers_and_object_access():
    count = get_settings().generation.question_template_count
    spec = TemplateSpecification(question_intent="Identify status", medical_constraints=[],
                                 execution_type="letter_classification", response_format="letter",
                                 template_slots=["modality"], language="en")
    for slot in ("answer", "modality.__class__", "target"):
        templates = [f"Question {index}: {{{slot}}}?" for index in range(count)]
        with pytest.raises(ValueError, match="slot"):
            validate_templates(spec, templates)
    templates = [f"Question {index}: locate {{target}}." for index in range(count)]
    assert validate_templates(spec.model_copy(update={"execution_type": "bbox_grounding", "response_format": "bbox", "template_slots": ["modality", "target"]}), templates) == templates




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
    options = extra.pop("options", ["benign", "malignant"] if response_format == "letter" else [])
    options = [re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", option).strip() for option in options]
    answer_field = extra.pop("answer_field", next((p for p in fields if ".bbox" in p), fields[0]))
    target_field = extra.pop("target_field", None)
    mapping = extra.pop("answer_value_mapping", [{"raw_value": o, "option": o} for o in options])
    return {
        "concept_id": "concept", "description": f"{response_format} task", "category": "test",
        "protocol": {
            "execution_type": {"letter": "letter_classification", "bbox": "bbox_grounding", "free_text": "free_text"}[response_format],
            "response_format": response_format,
            "metric": {"letter": "accuracy", "free_text": "bleu", "bbox": "miou"}[response_format],
            "data_binding": {"supporting_fields": fields, "answer_field": answer_field, "target_field": target_field},
            "answer_contract": {"options": options, "value_mapping": mapping},
            "question_intent": "Identify the annotated property", "medical_constraints": [],
            "template_slots": ["modality", "target"] if target_field else ["modality"],
        },
        "evidence": [], "question_templates": ["What is shown in this image?"], **extra,
    }


def _request(tmp_path: Path, records: list[dict], tasks: dict, target: int, seed: int = 0) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text(
        "".join((json.dumps(record) + "\n" for record in records)), encoding="utf-8"
    )
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    return {
        "benchmark_info": {"language": "en", "modality": "mammography"},
        "run_id": "00000000-0000-0000-0000-000000000000",
        "dataset_id": "test-dataset",
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


def _fake_model(calls):
    def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Batch instantiation must not call an LLM")

    return unexpected


def _candidate(task_id: str, annotation: dict, supporting_data: dict, task: dict):
    modality = annotation.pop("modality", None)
    record = _record(0, annotation)
    record["image"]["modality"] = modality
    expected_answer = None
    expected_evidence = None
    if task["protocol"]["response_format"] == "letter":
        expected_answer = task["protocol"]["answer_contract"]["options"][0]
        expected_evidence = {
            "path": task["protocol"]["data_binding"]["answer_field"],
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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ({}, None),
        ([], None),
        (float("nan"), None),
        ("benign", '"benign"'),
        (False, "false"),
        (1, "1"),
    ],
)
def test_answer_mapping_scalar_canonicalization(value: object, expected: str | None) -> None:
    assert canonical_answer_scalar(value) == expected


def test_question_template_selection_is_seeded_and_supports_single_template() -> None:
    task = _task("letter", ["image.annotation.status"])
    assert (
        generation._select_question_template(task, 7, "status", "img-1", 1)
        == task["question_templates"][0]
    )
    task["question_templates"] = ["one", "two", "three"]
    first = generation._select_question_template(task, 7, "status", "img-1", 1)
    assert generation._select_question_template(task, 7, "status", "img-1", 1) == first
    assert (
        len(
            {
                generation._select_question_template(task, 7, "status", f"img-{index}", index)
                for index in range(20)
            }
        )
        > 1
    )


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
    scalar = locate_scalar_evidence(record, ["image.annotation.report"], "Normal study.")
    assert scalar == {"path": "image.annotation.report", "indices": [], "value": "Normal study."}
    bbox = locate_bbox_evidence(
        record, ["image.annotation.findings[*].bbox[*]"], [10.0, 11.0, 20.0, 21.0]
    )
    assert [coordinate["indices"] for coordinate in bbox] == [[1, 0], [1, 1], [1, 2], [1, 3]]
    with pytest.raises(ValueError, match="exactly one"):
        locate_scalar_evidence(
            _record(0, {"first": "same", "second": "same"}),
            ["image.annotation.first", "image.annotation.second"],
            "same",
        )
    with pytest.raises(ValueError, match="valid xyxy"):
        locate_bbox_evidence(
            record, ["image.annotation.findings[*].bbox[*]"], [1, 2, float("inf"), 9]
        )


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
            [
                "image.annotation.findings[*].finding_categories[*]",
                "image.annotation.findings[*].bbox[*]",
            ],
            target_field="image.annotation.findings[*].finding_categories[*]",
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
    assert grounding.supports(unnamed_visual)  # ID alone cannot impose named-target behavior
    unnamed_visual.task["protocol"]["data_binding"]["target_field"] = "image.annotation.name"
    assert not grounding.supports(unnamed_visual)


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
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", fake)
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", fake)
    result = generation.generate_raw_benchmark(request)
    items = _read_items(result.raw_output_path)
    assert [item["task_id"] for item in items] == ["vqa", "grounding", "report"]
    assert not calls
    assert items[0]["answer"] == "A"
    assert items[1]["answer"] == [1.0, 2.0, 8.0, 9.0]
    assert items[2]["answer"] == "Normal study."
    assert all(
        (
            set(item)
            == {
                "id",
                "task_id",
                "dataset",
                "image_id",
                "image_path",
                "question",
                "options",
                "answer",
            }
            for item in items
        )
    )
    assert not list(Path(request["output_directory"]).glob("benchmark_items.*.progress.jsonl"))
    assert all(
        (temperature == 0.7 for prompt, temperature in calls if prompt.startswith("Generate one"))
    )
    assert all((temperature == 0 for prompt, temperature in calls if "Review one" in prompt))


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
            f"grounding_{index}": _task(
                "bbox", [f"image.annotation.bbox_{index}[*]"], sample_count=1
            )
            for index in range(4)
        },
    }
    request = _request(tmp_path, [_record(0, annotation)], tasks, 8)
    calls: list[tuple[str, float]] = []
    fake = _fake_model(calls)
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", fake)
    result = generation.generate_raw_benchmark(request)
    items = _read_items(result.raw_output_path)
    assert len([item for item in items if item["task_id"].startswith("vqa_")]) == 3
    assert len([item for item in items if item["task_id"].startswith("grounding_")]) == 3
    assert result.item_count == 6


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
    monkeypatch.setattr("medbenchagent.runtime.llm.complete_json", _fake_model([]))
    result = generation.generate_raw_benchmark(request)
    assert result.task_counts == {"vqa": 0, "grounding": 1}


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
    assert task["protocol"]["answer_contract"]["options"] == ["benign", "malignant"]
    task["question_templates"] = ["duplicate", "duplicate"]
    with pytest.raises(ValueError, match="unique non-empty"):
        generation._validate_tasks(request)
    assert not generation._all_supporting_fields_present(
        _record(0, {"status": "benign", "laterality": ""}), task["protocol"]["data_binding"]["supporting_fields"]
    )
    assert generation._all_supporting_fields_present(
        _record(0, {"status": "benign", "laterality": False}), task["protocol"]["data_binding"]["supporting_fields"]
    )


def test_letter_tasks_require_structured_answer_mapping() -> None:
    task = _task("letter", ["image.annotation.status"], sample_count=1)
    task["protocol"]["data_binding"].pop("answer_field")
    task["protocol"]["answer_contract"].pop("value_mapping")
    request = {
        "dataset_id": "test-dataset",
        "tasks": {"vqa": task},
        "target_sample_count": 1,
        "max_tasks_per_image": 3,
        "question_temperature": 0.7,
    }
    with pytest.raises(ValueError, match="answer_field"):
        generation._validate_tasks(request)


def test_generation_rejects_final_task_without_metric(tmp_path: Path) -> None:
    task = _task("letter", ["image.annotation.status"], sample_count=1)
    task["protocol"].pop("metric")
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"task": task}, 1)
    with pytest.raises(ValueError, match="metric"):
        generation.initialize_generation(request)


def test_generation_rejects_final_task_without_sample_count(tmp_path: Path) -> None:
    task = _task("letter", ["image.annotation.status"])
    request = _request(tmp_path, [_record(0, {"status": "benign"})], {"task": task}, 1)
    with pytest.raises(ValueError, match="sample_count"):
        generation.initialize_generation(request)


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
    monkeypatch.setattr(
        "medbenchagent.runtime.llm.complete_json", lambda *_args, **_kwargs: pytest.fail("model called")
    )
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
                "id": "existing",
                "task_id": "vqa",
                "dataset": "test-dataset",
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
    result = generation.finalize_generation(request)
    assert _read_items(result.raw_output_path)[0]["id"] == "existing"
    assert not progress.exists()


def _grounding(protocol):
    return [{"field": protocol["data_binding"]["answer_field"], "raw_value": m["raw_value"], "option": m["option"],
             "mode": "direct", "interpretation": "Preserve the explicit annotation label", "evidence_references": []}
            for m in protocol["answer_contract"]["value_mapping"]]
