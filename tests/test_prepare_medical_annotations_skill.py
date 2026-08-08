import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


SKILL_ROOT = (
    Path(__file__).parents[1] / ".codex" / "skills" / "prepare-medical-annotations"
)


def _validator() -> ModuleType:
    path = SKILL_ROOT / "scripts" / "validate_annotations.py"
    spec = importlib.util.spec_from_file_location("prepare_annotations_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(annotation: dict) -> dict:
    return {
        "patient": {"patient_id": None, "annotation": {}},
        "case": {"case_id": None, "annotation": {}},
        "study": {"study_id": None, "annotation": {}},
        "image": {
            "image_id": "image-1",
            "image_path": "images/image-1.png",
            "modality": None,
            "anatomy": None,
            "annotation": annotation,
        },
    }


def _validate(tmp_path: Path, annotation: dict) -> dict[str, object]:
    source_root = tmp_path / "source"
    image = source_root / "images" / "image-1.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text(json.dumps(_record(annotation)) + "\n", encoding="utf-8")
    return _validator().validate(
        annotation_path, SKILL_ROOT / "annotation.v2.schema.json", source_root
    )


def test_skill_validator_accepts_nested_canonical_xyxy(tmp_path: Path) -> None:
    result = _validate(
        tmp_path,
        {"findings": [{"bbox": [1, 2.5, 8, 9.25]}]},
    )

    assert result["records"] == 1


@pytest.mark.parametrize(
    "bbox",
    [
        {"xmin": 1, "ymin": 2, "xmax": 8, "ymax": 9},
        [1, 2, 8],
        [True, 2, 8, 9],
        [1, 2, float("inf"), 9],
        [-1, 2, 8, 9],
        [8, 2, 1, 9],
        [1, 9, 8, 2],
    ],
)
def test_skill_validator_rejects_invalid_bbox_values(
    tmp_path: Path, bbox: object
) -> None:
    with pytest.raises(ValueError, match="bbox"):
        _validate(tmp_path, {"finding": {"bbox": bbox}})


@pytest.mark.parametrize(
    "coordinates",
    [
        {"xmin": 1, "ymin": 2, "xmax": 8, "ymax": 9},
        {"x": 1, "y": 2, "width": 7, "height": 7},
        {"cx": 4.5, "cy": 5.5, "width": 7, "height": 7},
    ],
)
def test_skill_validator_rejects_unconverted_bbox_fields(
    tmp_path: Path, coordinates: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="non-canonical bbox coordinate fields"):
        _validate(tmp_path, {"finding": coordinates})
