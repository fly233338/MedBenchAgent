import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / ".codex" / "skills" / "medical-dataset-preparation" / "scripts" / "build_image_manifest.py"
SPEC = importlib.util.spec_from_file_location("image_manifest_skill", SCRIPT)
assert SPEC and SPEC.loader
manifest_skill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_skill)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _conversion(path: str = "source/a.dcm", status: str = "converted") -> dict:
    return {"source_relative_path": path, "status": status, "png_relative_path": "images/a.png", "width": 2, "height": 3, "output_sha256": "a" * 64}


def test_build_image_manifest_is_ordered_and_deterministic(tmp_path: Path) -> None:
    annotation, conversion = tmp_path / "annotation.jsonl", tmp_path / "conversion.jsonl"
    _jsonl(annotation, [{"image_id": "b", "image_path": "source/b.dcm"}, {"image_id": "a", "image_path": "source/a.dcm"}])
    _jsonl(conversion, [_conversion("source/a.dcm"), _conversion("source/b.dcm")])

    first = manifest_skill.build(annotation, conversion)
    second = manifest_skill.build(annotation, conversion)

    assert first == second
    assert [row["image_id"] for row in first] == ["b", "a"]
    assert first[0]["conversion_status"] == "success"


@pytest.mark.parametrize(
    ("annotation", "conversion", "message"),
    [
        ([{"image_id": "a", "image_path": "source/missing.dcm"}], [_conversion()], "No successful"),
        ([{"image_id": "a", "image_path": "source/a.dcm"}], [_conversion(status="failed")], "No successful"),
        ([{"image_id": "a", "image_path": "../a.dcm"}], [_conversion()], "traverse"),
        ([{"image_id": "a", "image_path": "source/a.dcm"}, {"image_id": "a", "image_path": "source/b.dcm"}], [_conversion(), _conversion("source/b.dcm")], "unique"),
    ],
)
def test_build_image_manifest_rejects_invalid_joins(tmp_path: Path, annotation: list[dict], conversion: list[dict], message: str) -> None:
    annotation_path, conversion_path = tmp_path / "annotation.jsonl", tmp_path / "conversion.jsonl"
    _jsonl(annotation_path, annotation)
    _jsonl(conversion_path, conversion)

    with pytest.raises(ValueError, match=message):
        manifest_skill.build(annotation_path, conversion_path)
