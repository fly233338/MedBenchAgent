import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "datasets"
    / "Normalized"
    / "ISIC2018"
    / "scripts"
    / "prepare_images.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("isic2018_prepare_images", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_images_copies_original_jpeg_into_artifact(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    source_root = tmp_path / "source"
    source_path = source_root / "images" / "image-one.jpg"
    source_path.parent.mkdir(parents=True)
    source = Image.new("RGB", (4, 3), (20, 40, 80))
    source.putpixel((0, 0), (255, 0, 0))
    source.save(source_path, format="JPEG")

    artifact = tmp_path / "artifact"
    annotation_path = artifact / "annotation.jsonl"
    artifact.mkdir(parents=True)
    annotation_path.write_text(
        json.dumps(
            {
                "patient": {"patient_id": None, "annotation": {}},
                "case": {"case_id": None, "annotation": {}},
                "study": {"study_id": None, "annotation": {}},
                "image": {
                    "image_id": "image-one",
                    "image_path": "images/image-one.jpg",
                    "modality": None,
                    "anatomy": None,
                    "annotation": {"width": 4, "height": 3},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image_output = artifact / "images"
    manifest = artifact / "image_manifest.jsonl"
    argv = [
        "prepare_images.py",
        "--annotation",
        str(annotation_path),
        "--source-root",
        str(source_root),
        "--image-output-dir",
        str(image_output),
        "--manifest-path",
        str(manifest),
        "--encoding",
        "original_jpeg",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert module.main() == 0
    copied = image_output / "image-one.jpg"
    row = json.loads(manifest.read_text(encoding="utf-8"))
    assert copied.read_bytes() == source_path.read_bytes()
    assert row["png_path"] == "images/image-one.jpg"
    assert row["sha256"] == module.sha256_file(copied)

    first_manifest = manifest.read_bytes()
    assert module.main() == 0
    assert manifest.read_bytes() == first_manifest
