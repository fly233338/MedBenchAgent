from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
sitk = pytest.importorskip("SimpleITK")
FileDataset = pydicom.dataset.FileDataset
FileMetaDataset = pydicom.dataset.FileMetaDataset
ExplicitVRLittleEndian = pydicom.uid.ExplicitVRLittleEndian
SecondaryCaptureImageStorage = pydicom.uid.SecondaryCaptureImageStorage
generate_uid = pydicom.uid.generate_uid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / ".codex"
    / "skills"
    / "convert-medical-images-to-png"
    / "scripts"
    / "convert_images.py"
)

if not SCRIPT.is_file():
    pytest.skip("optional local Codex image-conversion skill is not installed", allow_module_level=True)


def _load_module():
    spec = importlib.util.spec_from_file_location("convert_medical_images_skill", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert_images = _load_module()


def _write_dicom(path: Path, pixels, photometric: str = "MONOCHROME1") -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Synthetic"
    dataset.Rows = pixels.shape[0]
    dataset.Columns = pixels.shape[1]
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.WindowCenter = 100
    dataset.WindowWidth = 200
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    dataset.save_as(path)


def _write_rgb_dicom(path: Path) -> None:
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8
    )
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape[:2]
    dataset.PhotometricInterpretation = "RGB"
    dataset.PlanarConfiguration = 0
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 3
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path)


def test_converter_supports_declared_2d_formats_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("L", (8, 6), 100).save(source / "jpeg-with-wrong-extension.bin", format="JPEG")
    Image.fromarray(np.arange(48, dtype=np.uint16).reshape(6, 8)).save(source / "high.tiff")
    _write_dicom(source / "dicom-without-extension", np.arange(48).reshape(6, 8))
    _write_rgb_dicom(source / "rgb.dcm")
    nib.save(
        nib.Nifti1Image(np.arange(48, dtype=np.int16).reshape(6, 8), np.eye(4)),
        source / "scan.nii.gz",
    )
    itk_image = sitk.GetImageFromArray(np.arange(48, dtype=np.int16).reshape(6, 8))
    sitk.WriteImage(itk_image, source / "scan.nrrd")
    sitk.WriteImage(itk_image, source / "scan.mha")
    sitk.WriteImage(itk_image, source / "scan.mhd")

    output = tmp_path / "converted"
    report = convert_images.convert(source, output)
    first_manifest = (output / "image_conversion_manifest.jsonl").read_bytes()
    repeated = convert_images.convert(source, output)

    rows = [json.loads(line) for line in first_manifest.decode().splitlines()]
    assert report["converted_count"] == 8
    assert report["skipped_unsupported_count"] == 1
    assert repeated["reused_identical_output"] is True
    assert (output / "image_conversion_manifest.jsonl").read_bytes() == first_manifest
    converted_rows = [row for row in rows if row["status"] == "converted"]
    assert len(converted_rows) == 8
    assert {row["detected_format"] for row in converted_rows} >= {
        "dicom",
        "itk",
        "nifti",
        "raster:jpeg",
        "raster:tiff",
    }
    for row in converted_rows:
        with Image.open(output / row["png_relative_path"]) as image:
            assert image.mode == "RGB"
            assert image.size == (8, 6) or image.size == (2, 2)


def test_converter_rejects_volume_and_leaves_no_partial_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (4, 4), "white").save(source / "valid.png")
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 2), dtype=np.float32), np.eye(4)),
        source / "volume.nii.gz",
    )
    output = tmp_path / "converted"

    with pytest.raises(convert_images.ConversionError, match="must squeeze to 2D"):
        convert_images.convert(source, output)
    assert not output.exists()


def test_converter_rejects_multiframe_dicom_and_wsi(tmp_path: Path) -> None:
    multiframe_source = tmp_path / "multiframe"
    multiframe_source.mkdir()
    path = multiframe_source / "multi.dcm"
    pixels = np.zeros((2, 4, 4), dtype=np.uint16)
    _write_dicom(path, pixels[0])
    dataset = pydicom.dcmread(path)
    dataset.NumberOfFrames = 2
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path)
    with pytest.raises(convert_images.ConversionError, match="multi-frame DICOM"):
        convert_images.convert(multiframe_source, tmp_path / "multi-output")
    assert not (tmp_path / "multi-output").exists()

    wsi_source = tmp_path / "wsi"
    wsi_source.mkdir()
    (wsi_source / "slide.svs").write_bytes(b"not-a-real-slide")
    with pytest.raises(convert_images.ConversionError, match="whole-slide image"):
        convert_images.convert(wsi_source, tmp_path / "wsi-output")
    assert not (tmp_path / "wsi-output").exists()


def test_converter_decodes_compressed_dicom(tmp_path: Path) -> None:
    source_file = pydicom.data.get_testdata_file("JPEG2000.dcm", download=False)
    if source_file is None:
        pytest.skip("pydicom JPEG2000 fixture is unavailable")
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(source_file, source / "compressed-without-extension")

    output = tmp_path / "converted"
    report = convert_images.convert(source, output)

    assert report["converted_count"] == 1
    row = json.loads(
        (output / "image_conversion_manifest.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["detected_format"] == "dicom"
    with Image.open(output / row["png_relative_path"]) as image:
        assert image.mode == "RGB"
