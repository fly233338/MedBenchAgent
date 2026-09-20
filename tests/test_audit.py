from test_generation import _task
from medbenchagent.data.image_assets import validate_image_asset
from PIL import Image
from medbenchagent.audit.hard import _answer_matches, _contract


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
    assert _contract(dict(item(), options=None, answer=[0, 0, 5, 5]), _task("bbox", ["image.annotation.bbox[*]"]), (2, 2)) == [
        "bbox"
    ]



def test_constant_image_rejected(tmp_path):
    Image.new("L", (2, 2)).save(tmp_path / "im.png")
    assert validate_image_asset({"png_path": "im.png", "width": 2, "height": 2}, tmp_path) == [
        "image_pixels"
    ]



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


