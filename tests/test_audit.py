from test_generation import _task
from medbenchagent.data.image_assets import validate_image_asset
import json
from pathlib import Path
from PIL import Image
from medbenchagent.audit.hard import _answer_matches, _contract, audit_run
from medbenchagent.audit.semantic_audit import semantic_audit_run
from medbenchagent.runtime.llm import ModelUnavailable


def record():
    return {
        "patient": {"patient_id": None, "annotation": {}},
        "case": {"case_id": None, "annotation": {}},
        "study": {"study_id": None, "annotation": {}},
        "image": {
            "image_id": "im",
            "image_path": "im.png",
            "modality": "US",
            "anatomy": "breast",
            "annotation": {"label": "benign", "report": "Normal.", "bbox": [0, 0, 1, 1]},
        },
    }


def task():
    return _task("letter", ["image.annotation.label"])


def item():
    return {
        "id": "one",
        "task_id": "view",
        "dataset": "set",
        "image_id": "im",
        "image_path": "im.png",
        "question": "Which category?",
        "options": ["benign", "malignant"],
        "answer": "A",
    }


def test_answers_checked_against_unique_annotation():
    assert _answer_matches(item(), task(), record())
    assert not _answer_matches(dict(item(), answer="B"), task(), record())
    assert _answer_matches(
        dict(item(), answer="Normal."),
        _task("free_text", ["image.annotation.report"]),
        record(),
    )
    assert _answer_matches(
        dict(item(), task_id="detection", answer=[0, 0, 1, 1]),
        _task("bbox", ["image.annotation.bbox[*]"]),
        record(),
    )


def test_invalid_options_and_dimensions():
    assert _contract(dict(item(), options=[{}]), task(), None) == ["letter_answer"]
    assert _contract(dict(item(), answer=[0, 0, 5, 5]), _task("bbox", ["image.annotation.bbox[*]"]), (2, 2)) == [
        "bbox"
    ]


def test_audit_rejects_wrong_answers_and_duplicate_ids(tmp_path):
    image = Image.new("L", (2, 2))
    image.putpixel((0, 0), 255)
    image.save(tmp_path / "im.png")
    files = {
        "blueprint.json": {"dataset": {"dataset_id": "set"}, "tasks": {"view": task()}},
        "annotation.jsonl": record(),
        "image_manifest.jsonl": {"image_id": "im", "png_path": "im.png", "width": 2, "height": 2},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
    (tmp_path / "raw.jsonl").write_text(
        "\n".join(json.dumps(i) for i in (item(), item(), dict(item(), id="wrong", answer="B"))),
        encoding="utf-8",
    )
    outputs = audit_run(
        raw_path=tmp_path / "raw.jsonl",
        blueprint_path=tmp_path / "blueprint.json",
        events_path=tmp_path / "events.jsonl",
        manifest_path=tmp_path / "image_manifest.jsonl",
        annotation_path=tmp_path / "annotation.jsonl",
        output_directory=tmp_path,
        run_id="run",
    )
    report = json.loads(Path(outputs["report_path"]).read_text())
    assert report["hard_audit"]["accepted"] == 1
    assert report["hard_audit"]["rejected"] == 2
    assert set(outputs) == {"audited_path", "report_path", "item_count"}


def test_constant_image_rejected(tmp_path):
    Image.new("L", (2, 2)).save(tmp_path / "im.png")
    assert validate_image_asset({"png_path": "im.png", "width": 2, "height": 2}, tmp_path) == [
        "image_pixels"
    ]


def test_semantic_failure_remains_observational(tmp_path, monkeypatch):
    (tmp_path / "audited.jsonl").write_text(json.dumps(item()) + "\n")
    (tmp_path / "blueprint.json").write_text(json.dumps({"tasks": {"view": task()}}))
    (tmp_path / "report.json").write_text(json.dumps({"hard_audit": {"accepted": 1}}))

    def fail(*args, **kwargs):
        raise ModelUnavailable("offline")

    monkeypatch.setattr("medbenchagent.audit.semantic_audit.complete_json", fail)
    semantic_audit_run(
        audited_path=tmp_path / "audited.jsonl",
        report_path=tmp_path / "report.json",
        blueprint_path=tmp_path / "blueprint.json",
        events_path=tmp_path / "events.jsonl",
        output_directory=tmp_path,
        run_id="run",
        dataset_id="set",
        seed=0,
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["semantic_audit"]["not_run"] == 1
    assert report["hard_audit"]["accepted"] == 1
    assert len(report["semantic_records"]) == 1


def test_shared_image_validation_preserves_consumer_policies(tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (2, 2), "black").save(path)
    entry = {"png_path": path.name, "width": 2, "height": 2}
    assert validate_image_asset(entry, tmp_path) == ["image_pixels"]
    assert validate_image_asset(
        entry, tmp_path, check_uniform=False, supported_formats=frozenset({"PNG", "JPEG"})
    ) == []
    Image.new("RGB", (2, 2), "black").save(path, format="BMP")
    assert validate_image_asset(
        entry, tmp_path, check_uniform=False, supported_formats=frozenset({"PNG", "JPEG"})
    ) == ["image_format"]
    assert validate_image_asset(dict(entry, width=3), tmp_path) == ["image_dimensions"]
