"""Small image readers with a separate header-only inspection boundary."""

from pathlib import Path


class ReaderUnavailable(RuntimeError):
    """Required decoder or reader dependency is missing, not a corrupt sample."""


class ImageReader:
    format = ""

    def probe(self, path):
        from medbenchagent.data.image_conversion import detect_format

        return (detect_format(Path(path)) or "").startswith(self.format)

    def read_pixels(self, path):
        raise NotImplementedError

    def convert(self, path, policy=None):
        from medbenchagent.data.image_conversion import convert_image

        return convert_image(path, policy)


class RasterReader(ImageReader):
    format = "raster:"

    def read_pixels(self, path):
        from PIL import Image
        from medbenchagent.data.image_conversion import UnsupportedMedicalImage

        with Image.open(path) as image:
            if (
                getattr(image, "n_frames", 1) != 1
                or image.format == "TIFF"
                and image.tag_v2.get(330)
            ):
                raise UnsupportedMedicalImage("Multiframe or pyramidal raster is unsupported")
            image.load()
            return image.copy(), "raster:" + image.format.lower(), None, False

    def read_metadata(self, path):
        from PIL import Image

        with Image.open(path) as image:
            return {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "frames": getattr(image, "n_frames", 1),
            }


DICOM_TAGS = [
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "ViewPosition",
    "Laterality",
    "ImageLaterality",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "WindowCenter",
    "WindowWidth",
    "VOILUTFunction",
    "RescaleSlope",
    "RescaleIntercept",
    "PixelSpacing",
]


class DicomReader(ImageReader):
    format = "dicom"

    def read_metadata(self, path):
        try:
            from pydicom import dcmread
        except ImportError as exc:
            raise ReaderUnavailable("Install pydicom for DICOM inspection") from exc
        dataset = dcmread(path, stop_before_pixels=True, specific_tags=DICOM_TAGS)

        def plain(value):
            if isinstance(value, (str, int, float)):
                return value
            if hasattr(value, "__iter__"):
                return [plain(v) for v in value]
            return str(value)

        result = {tag: plain(getattr(dataset, tag)) for tag in DICOM_TAGS if hasattr(dataset, tag)}
        result["TransferSyntaxUID"] = str(getattr(dataset.file_meta, "TransferSyntaxUID", ""))
        return result

    def read_pixels(self, path):
        from medbenchagent.data.image_conversion import UnsupportedMedicalImage

        try:
            from pydicom import dcmread
            from pydicom.pixels import apply_modality_lut
        except ImportError as exc:
            raise ReaderUnavailable("Install pydicom and its pixel decoder dependencies") from exc
        dataset = dcmread(path)
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise UnsupportedMedicalImage("Multiframe DICOM is unsupported")
        try:
            pixels = apply_modality_lut(dataset.pixel_array, dataset)
        except RuntimeError as exc:
            if "plugin" in str(exc).lower() or "dependencies" in str(exc).lower():
                raise ReaderUnavailable(f"DICOM decoder unavailable: {exc}") from exc
            raise
        interpretation = str(getattr(dataset, "PhotometricInterpretation", ""))
        if interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
            raise UnsupportedMedicalImage("Color DICOM requires a documented color conversion")
        center, width = (
            getattr(dataset, "WindowCenter", None),
            getattr(dataset, "WindowWidth", None),
        )
        window = None
        if (
            isinstance(center, (int, float))
            and isinstance(width, (int, float))
            and float(width) > 1
        ):
            if str(getattr(dataset, "VOILUTFunction", "LINEAR")) == "LINEAR":
                window = [
                    float(center) - 0.5 - (float(width) - 1) / 2,
                    float(center) - 0.5 + (float(width) - 1) / 2,
                ]
        return _plane(pixels), self.format, window, interpretation == "MONOCHROME1"


class NiftiReader(ImageReader):
    format = "nifti"

    def read_pixels(self, path):
        import numpy as np

        try:
            import nibabel as nib
        except ImportError as exc:
            raise ReaderUnavailable("Install nibabel") from exc
        return _plane(np.squeeze(np.asanyarray(nib.load(path).dataobj))), self.format, None, False

    def read_metadata(self, path):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ReaderUnavailable("Install nibabel") from exc
        image = nib.load(path)
        return {
            "shape": list(image.shape),
            "dtype": str(image.get_data_dtype()),
            "affine": image.affine.tolist(),
        }


class ItkReader(ImageReader):
    format = "itk"

    def read_pixels(self, path):
        import numpy as np

        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ReaderUnavailable("Install SimpleITK") from exc
        return (
            _plane(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))),
            self.format,
            None,
            False,
        )

    def read_metadata(self, path):
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ReaderUnavailable("Install SimpleITK") from exc
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return {
            "size": list(reader.GetSize()),
            "spacing": list(reader.GetSpacing()),
            "direction": list(reader.GetDirection()),
            "dimension": reader.GetDimension(),
        }


READERS = (RasterReader(), DicomReader(), NiftiReader(), ItkReader())


def _plane(pixels):
    from medbenchagent.data.image_conversion import UnsupportedMedicalImage

    if pixels.ndim != 2:
        raise UnsupportedMedicalImage(f"Only 2D images are supported, got {pixels.shape}")
    return pixels


def reader_for(path):
    from medbenchagent.data.image_conversion import UnsupportedMedicalImage

    for reader in READERS:
        if reader.probe(path):
            return reader
    raise UnsupportedMedicalImage(f"Unsupported medical image: {Path(path).name}")


def inspect_metadata(path):
    return reader_for(path).read_metadata(path)
