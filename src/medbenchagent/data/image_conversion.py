"""Content detection and explicit, geometry-preserving 2D image conversion."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

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
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format in RASTER_FORMATS:
                return f"raster:{image.format.lower()}"
    except (OSError, ValueError):
        pass
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
    return None


def _pixels(path):
    import numpy as np
    from PIL import Image

    detected = detect_format(path)
    window = None
    inverse = False
    if detected and detected.startswith("raster:"):
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1 or (
                image.format == "TIFF" and image.tag_v2.get(330)
            ):
                raise UnsupportedMedicalImage("Multiframe or pyramidal raster is unsupported")
            image.load()
            return image.copy(), detected, window, inverse
    if detected == "dicom":
        from pydicom import dcmread
        from pydicom.pixels import apply_modality_lut

        dataset = dcmread(path)
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise UnsupportedMedicalImage("Multiframe DICOM is unsupported")
        pixels = apply_modality_lut(dataset.pixel_array, dataset)
        if str(getattr(dataset, "PhotometricInterpretation", "")) not in {
            "MONOCHROME1",
            "MONOCHROME2",
        }:
            raise UnsupportedMedicalImage("Color DICOM requires a documented color conversion")
        inverse = str(dataset.PhotometricInterpretation) == "MONOCHROME1"
        center, width = (
            getattr(dataset, "WindowCenter", None),
            getattr(dataset, "WindowWidth", None),
        )
        # A single linear window is source-supported; never choose among multiple windows.
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
    elif detected == "nifti":
        import nibabel as nib

        pixels = np.squeeze(np.asanyarray(nib.load(path).dataobj))
    elif detected == "itk":
        import SimpleITK as sitk

        pixels = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(str(path))))
    else:
        raise UnsupportedMedicalImage(f"Unsupported image: {path.name}")
    if pixels.ndim != 2:
        raise UnsupportedMedicalImage(f"Only 2D images are supported, got {pixels.shape}")
    return pixels, detected, window, inverse


def inspect_image(path):
    """Return format/geometry and any source-supported display window, without transforming pixels."""
    import numpy as np
    from PIL import Image

    pixels, detected, window, inverse = _pixels(Path(path))
    if isinstance(pixels, Image.Image):
        size, mode = pixels.size, pixels.mode
        safe = mode in {"1", "L", "LA", "RGB", "RGBA", "P", "I;16"}
    else:
        size, mode = (pixels.shape[1], pixels.shape[0]), str(pixels.dtype)
        safe = pixels.dtype in (np.dtype("uint8"), np.dtype("uint16")) and not inverse
    return {
        "format": detected,
        "width": size[0],
        "height": size[1],
        "mode": mode,
        "copy": detected in {"raster:png", "raster:jpeg"},
        "preserve_supported": safe,
        "window": window,
        "invert": inverse,
    }


def convert_image(path, policy=None):
    """Preserve representable pixels or apply an explicitly selected linear display window."""
    import numpy as np
    from PIL import Image

    policy = policy or {"encoding": "preserve"}
    pixels, detected, window, inverse = _pixels(Path(path))
    info = inspect_image(path)
    if policy["encoding"] == "preserve":
        if not info["preserve_supported"]:
            raise ConversionError(
                "A documented display window or explicit user decision is required"
            )
        image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
    elif policy["encoding"] == "window":
        low, high = policy["low"], policy["high"]
        if not np.isfinite([low, high]).all() or low >= high:
            raise ConversionError("Display window requires finite low < high")
        values = np.asarray(pixels, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ConversionError("Windowing requires finite scalar 2D pixels")
        rendered = np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        if policy.get("invert", False):
            rendered = 255 - rendered
        image = Image.fromarray(rendered)
    else:
        raise ConversionError("Unknown image encoding")
    return image, detected, policy["encoding"]


def write_image(source_path, output_path, policy=None):
    """Copy PNG/JPEG bytes; write other supported 2D inputs as PNG without spatial changes."""
    import shutil

    source_path, output_path = Path(source_path), Path(output_path)
    info = inspect_image(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if info["copy"]:
        shutil.copyfile(source_path, output_path)
    else:
        image, _, _ = convert_image(source_path, policy)
        image.save(output_path, format="PNG", compress_level=9)
    return info
