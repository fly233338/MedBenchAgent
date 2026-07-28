"""Reference functions for converting one 2D medical image to grayscale PNG.

Copy only the handlers needed by the current dataset into its generated
conversion script. Expected packages are numpy, Pillow, pydicom with the
required pixel decoders, nibabel, and SimpleITK.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

WSI_SUFFIXES = {".bif", ".mrxs", ".ndpi", ".scn", ".svs"}
NIFTI_SUFFIXES = {".nii"}
ITK_SUFFIXES = {".mha", ".mhd", ".nhdr", ".nrrd"}
RASTER_FORMATS = {"BMP", "JPEG", "PNG", "TIFF", "WEBP"}


class ConversionError(RuntimeError):
    """Raised when an image cannot be converted safely."""


class UnsupportedMedicalImage(ConversionError):
    """Raised when an image is intentionally outside this workflow."""


def _read_header(path: Path, size: int = 560) -> bytes:
    with path.open("rb") as source:
        prefix = source.read(2)
        source.seek(0)
        if prefix == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=source) as compressed:
                return compressed.read(size)
        return source.read(size)


def _looks_like_nifti(path: Path) -> bool:
    try:
        header = _read_header(path)
    except (OSError, EOFError):
        return False
    if len(header) >= 348:
        sizes = {struct.unpack("<I", header[:4])[0], struct.unpack(">I", header[:4])[0]}
        if 348 in sizes and header[344:348] in {b"n+1\x00", b"ni1\x00"}:
            return True
    if len(header) >= 12:
        sizes = {struct.unpack("<I", header[:4])[0], struct.unpack(">I", header[:4])[0]}
        if 540 in sizes and header[4:8] in {b"n+2\x00", b"ni2\x00"}:
            return True
    return False


def _looks_like_itk(path: Path) -> bool:
    try:
        header = _read_header(path, 256)
    except (OSError, EOFError):
        return False
    return header.startswith(b"NRRD000") or b"ObjectType = Image" in header


def _looks_like_dicom(path: Path) -> bool:
    try:
        from pydicom import dcmread
        from pydicom.misc import is_dicom

        if is_dicom(path):
            return True
        dataset = dcmread(
            path,
            stop_before_pixels=True,
            force=True,
            specific_tags=["Rows", "Columns", "SOPClassUID", "PhotometricInterpretation"],
        )
        return bool(
            getattr(dataset, "Rows", None)
            and getattr(dataset, "Columns", None)
            and getattr(dataset, "SOPClassUID", None)
        )
    except Exception:
        return False


def detect_format(path: Path) -> str | None:
    """Detect a supported medical image format by suffix and file content."""
    lower_name = path.name.lower()
    if path.suffix.lower() in WSI_SUFFIXES:
        raise UnsupportedMedicalImage(f"whole-slide image is unsupported: {path.name}")
    if lower_name.endswith(".nii.gz") or path.suffix.lower() in NIFTI_SUFFIXES:
        return "nifti"
    if path.suffix.lower() in ITK_SUFFIXES:
        return "itk"
    if _looks_like_nifti(path):
        return "nifti"
    if _looks_like_itk(path):
        return "itk"
    if _looks_like_dicom(path):
        return "dicom"
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format in RASTER_FORMATS:
                return f"raster:{image.format.lower()}"
    except Exception:
        return None
    return None


def _normalize_numeric(array: Any) -> tuple[Any, str]:
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        raise ConversionError("image contains no finite pixel values")
    finite_values = values[finite]
    low, high = np.percentile(finite_values, [0.5, 99.5])
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8), "constant_to_zero"
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.rint(scaled * 255.0).astype(np.uint8), "percentile_0.5_99.5"


def _scalar_image(array: Any) -> tuple[Any, str]:
    from PIL import Image

    gray, normalization = _normalize_numeric(array)
    return Image.fromarray(gray), normalization


def _grayscale_color_image(array: Any) -> tuple[Any, str]:
    """Accept grayscale stored in color channels and reject genuine color."""
    import numpy as np
    from PIL import Image

    values = np.asarray(array)
    if values.ndim != 3 or values.shape[-1] not in {3, 4}:
        raise ConversionError(f"expected HxWx3/4 color pixels, got {values.shape}")
    if values.dtype == np.uint8:
        rendered = values
        normalization = "preserved_uint8_channels"
    else:
        rendered, normalization = _normalize_numeric(values)
    red, green, blue = rendered[..., 0], rendered[..., 1], rendered[..., 2]
    if not (np.array_equal(red, green) and np.array_equal(red, blue)):
        raise UnsupportedMedicalImage("genuinely color images require explicit user direction")
    gray = red
    if rendered.shape[-1] == 4:
        alpha = rendered[..., 3].astype(np.float64) / 255.0
        gray = np.rint(gray.astype(np.float64) * alpha).astype(np.uint8)
        normalization += "+alpha_on_black"
    return Image.fromarray(gray), normalization


def _convert_raster(path: Path) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise UnsupportedMedicalImage(f"multi-page raster is unsupported: {path.name}")
        if source.format == "TIFF" and source.tag_v2.get(330):
            raise UnsupportedMedicalImage(f"pyramidal TIFF is unsupported: {path.name}")
        source.load()
        if source.mode in {"I", "I;16", "I;16B", "I;16L", "F"}:
            return _scalar_image(np.asarray(source))
        if source.mode in {"1", "L"}:
            return source.convert("L"), "preserved_display_values"
        has_alpha = source.mode in {"LA", "RGBA"} or "transparency" in source.info
        rendered = source.convert("RGBA") if has_alpha else source.convert("RGB")
        return _grayscale_color_image(np.asarray(rendered))


def _convert_dicom(path: Path) -> tuple[Any, str]:
    import numpy as np
    from PIL import Image
    from pydicom import dcmread
    from pydicom.pixels import (
        apply_color_lut,
        apply_modality_lut,
        apply_voi_lut,
        convert_color_space,
    )

    dataset = dcmread(path)
    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if frames != 1:
        raise UnsupportedMedicalImage(f"multi-frame DICOM is unsupported: {path.name}")
    if hasattr(dataset, "pixel_array_options"):
        dataset.pixel_array_options(raw=True)
    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", "unknown")
        raise ConversionError(f"cannot decode DICOM transfer syntax {syntax}: {exc}") from exc

    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric == "PALETTE COLOR":
        image, normalization = _grayscale_color_image(apply_color_lut(pixels, dataset))
        return image, f"dicom_palette_color_lut+{normalization}"
    if photometric in {"RGB", "YBR_FULL", "YBR_FULL_422"}:
        color = np.asarray(pixels)
        normalization = "dicom_rgb"
        if photometric.startswith("YBR"):
            color = convert_color_space(color, photometric, "RGB")
            normalization = "dicom_ybr_to_rgb"
        image, channel_normalization = _grayscale_color_image(color)
        return image, f"{normalization}+{channel_normalization}"
    if np.asarray(pixels).ndim != 2:
        raise ConversionError(f"unsupported DICOM pixel shape: {np.asarray(pixels).shape}")

    transformed = apply_modality_lut(pixels, dataset)
    steps = ["dicom_modality_lut_or_rescale"]
    has_voi = hasattr(dataset, "VOILUTSequence") or (
        hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth")
    )
    if has_voi:
        transformed = apply_voi_lut(transformed, dataset, index=0)
        steps.append("voi_lut_or_window_0")
    gray, normalization = _normalize_numeric(transformed)
    steps.append(normalization)
    inverse = (
        photometric == "MONOCHROME1"
        or str(getattr(dataset, "PresentationLUTShape", "")).upper() == "INVERSE"
    )
    if inverse:
        gray = 255 - gray
        steps.append("inverse")
    return Image.fromarray(gray), "+".join(steps)


def _convert_nifti(path: Path) -> tuple[Any, str]:
    import nibabel as nib
    import numpy as np

    image = nib.load(path)
    data = np.squeeze(image.get_fdata(dtype=np.float32, caching="unchanged"))
    if data.ndim != 2:
        raise UnsupportedMedicalImage(f"NIfTI must squeeze to 2D, got {data.shape}: {path.name}")
    converted, normalization = _scalar_image(data)
    return converted, f"nifti_scaled+{normalization}"


def _convert_itk(path: Path) -> tuple[Any, str]:
    import numpy as np
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    data = np.squeeze(sitk.GetArrayFromImage(image))
    if data.ndim != 2:
        raise UnsupportedMedicalImage(f"ITK image must squeeze to 2D: {path.name}")
    converted, normalization = _scalar_image(data)
    return converted, f"itk_scaled+{normalization}"


def convert_image(path: Path) -> tuple[Any, str, str]:
    """Return a grayscale Pillow image, detected format, and normalization."""
    detected = detect_format(path)
    if detected is None:
        raise ConversionError(f"unrecognized image format: {path.name}")
    if detected.startswith("raster:"):
        image, normalization = _convert_raster(path)
    elif detected == "dicom":
        image, normalization = _convert_dicom(path)
    elif detected == "nifti":
        image, normalization = _convert_nifti(path)
    else:
        image, normalization = _convert_itk(path)
    if image.mode != "L":
        raise ConversionError(f"conversion did not produce grayscale pixels: {path.name}")
    return image, detected, normalization


def write_png(source_path: Path, output_path: Path) -> dict[str, object]:
    """Convert one source image and write a deterministic grayscale PNG."""
    image, detected, normalization = convert_image(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", compress_level=9)
    return {
        "detected_format": detected,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "normalization": normalization,
    }
