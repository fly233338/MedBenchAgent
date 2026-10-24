import numpy as np
import pytest

from medbenchagent.data.asset_adapters import AssetAdapter, LoadedAsset
from medbenchagent.data.models import AssetEntry, PresentationSpec


class _Volume(AssetAdapter):
    kind = "volume"
    formats = frozenset({"test"})

    def load(self, asset_spec, *, source_root=None):
        return LoadedAsset(np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4), {})


def test_presentation_applies_window_crop_resize_before_slicing():
    adapter = _Volume()
    asset = AssetEntry(asset_id="v", kind="volume", format="test", source_path="v", file_digest="d")
    spec = PresentationSpec(
        mode="slices",
        axis=0,
        indices=[1],
        window=(10, 40),
        crop=(0, 0, 1, 1),
        resize=(2, 2),
    )
    result = adapter.present(asset, spec)
    assert result.data.shape == (1, 2, 2)
    assert result.data.dtype == np.uint8


@pytest.mark.parametrize("format", ["nifti", "nrrd", "dicom", "dicom_series"])
def test_real_medical_readers_materialize_png(tmp_path, format):
    from medbenchagent.data.sources import digest_path
    from medbenchagent.data.asset_adapters import materialize_image

    from PIL import Image

    source = tmp_path / "raw"
    source.mkdir()
    data = np.arange(4 * 4 * 3, dtype=np.uint16).reshape(4, 4, 3)
    if format == "nifti":
        nib = pytest.importorskip("nibabel")
        path = source / "volume.nii.gz"
        affine = np.eye(4)
        affine[:3, 3] = [11, -7, 3]
        nib.save(nib.Nifti1Image(data, affine), path)
    elif format == "nrrd":
        nrrd = pytest.importorskip("nrrd")
        path = source / "volume.nrrd"
        nrrd.write(str(path), data)
    else:
        pydicom = pytest.importorskip("pydicom")
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        path = source / ("series" if format == "dicom_series" else "image.dcm")
        if format == "dicom_series":
            path.mkdir()
        series, study = generate_uid(), generate_uid()
        for index in range(3 if format == "dicom_series" else 1):
            destination = path / f"{index}.dcm" if path.is_dir() else path
            meta = pydicom.dataset.FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = pydicom.dataset.FileDataset(
                str(destination),
                {},
                file_meta=meta,
                preamble=b"\0" * 128,
            )
            ds.SOPClassUID, ds.SOPInstanceUID = (
                meta.MediaStorageSOPClassUID,
                meta.MediaStorageSOPInstanceUID,
            )
            ds.SeriesInstanceUID, ds.StudyInstanceUID, ds.PatientID = series, study, "test"
            ds.Rows, ds.Columns, ds.SamplesPerPixel = 4, 4, 1
            ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
            ds.PhotometricInterpretation = "MONOCHROME1"
            ds.RescaleSlope, ds.RescaleIntercept = 2, -10
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, index]
            ds.InstanceNumber = index + 1
            ds.PixelData = data[:, :, index].copy().tobytes()
            ds.save_as(destination, enforce_file_format=True)
    asset = AssetEntry(
        asset_id="medical",
        source_id="s",
        source_path=path.relative_to(source).as_posix(),
        kind="image" if format == "dicom" else "volume",
        format=format,
        file_digest=digest_path(path),
    )
    spec = (
        PresentationSpec(mode="native", window=(-10, 100))
        if format == "dicom"
        else PresentationSpec(
            mode="slices",
            axis=0 if format == "dicom_series" else 2,
            indices=[1],
            window=(-10, 100),
        )
    )
    output = tmp_path / "display.png"
    info = materialize_image(asset, spec, output, source_root=source)
    assert info["format"] == "PNG"
    with Image.open(output) as image:
        assert image.size == (4, 4)
        plane = data[:, :, 0 if format == "dicom" else 1].astype(float)
        if format.startswith("dicom"):
            plane = plane * 2 - 10
        expected = np.rint(np.clip((plane + 10) / 110, 0, 1) * 255).astype(np.uint8)
        if format.startswith("dicom"):
            expected = 255 - expected
        np.testing.assert_array_equal(np.asarray(image), expected)

    if format == "nifti":
        from medbenchagent.data.asset_adapters import NiftiVolumeAdapter

        assert NiftiVolumeAdapter().load(asset, source_root=source).metadata["origin"] == [
            11,
            -7,
            3,
        ]
    if format == "dicom_series":
        from medbenchagent.data.asset_adapters import DicomSeriesAdapter, AssetAdapterError

        (path / "2.dcm").unlink()
        ds = pydicom.dcmread(path / "1.dcm")
        ds.ImagePositionPatient = [0, 0, 0]
        ds.save_as(path / "1.dcm", enforce_file_format=True)
        asset = asset.model_copy(update={"file_digest": digest_path(path)})
        with pytest.raises(AssetAdapterError, match="duplicate slice"):
            DicomSeriesAdapter().load(asset, source_root=source)
