"""Native Asset readers and deterministic presentation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from .sources import digest_file
from typing import Any

import numpy as np
import gzip
import struct

from medbenchagent.data.models import AssetEntry, PresentationSpec


class AssetAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedAsset:
    data: Any
    metadata: dict[str, Any]


class AssetAdapter:
    kind: str
    formats: frozenset[str]

    def inspect(self, asset_spec: AssetEntry, *, source_root: str | Path | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, asset_spec: AssetEntry, *, source_root: str | Path | None = None) -> list[str]:
        try:
            self.inspect(asset_spec, source_root=source_root)
        except Exception as exc:
            return [str(exc)]
        return []

    def load(self, asset_spec: AssetEntry, *, source_root: str | Path | None = None) -> LoadedAsset:
        raise NotImplementedError

    def present(self, asset_spec: AssetEntry, presentation_spec: PresentationSpec, *, source_root: str | Path | None = None) -> LoadedAsset:
        loaded = self.load(asset_spec, source_root=source_root)
        if presentation_spec.mode == "native":
            from PIL import Image
            data = loaded.data
            if presentation_spec.window:
                low, high = presentation_spec.window
                policy = {"encoding": "window", "low": low, "high": high,
                          "invert": loaded.metadata.get("invert", False)}
            else:
                policy = {"encoding": "preserve"}
            data, _, _ = _render_pixels(policy=policy,
                decoded=(data, asset_spec.format, loaded.metadata.get("window"), loaded.metadata.get("invert", False)))
            if presentation_spec.crop:
                x0, y0, x1, y1 = presentation_spec.crop
                width, height = data.size
                data = data.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
            if presentation_spec.resize:
                data = data.resize(presentation_spec.resize, Image.Resampling.NEAREST)
            return LoadedAsset(data, loaded.metadata)
        data = np.asarray(loaded.data)
        if data.ndim < 3:
            raise AssetAdapterError("slice presentation requires a volume")
        if data.ndim > 3:
            raise AssetAdapterError("4D volume requires an explicit frame extraction before presentation")
        axis = presentation_spec.axis if presentation_spec.axis is not None else 0
        if axis < 0 or axis >= data.ndim:
            raise AssetAdapterError(f"Presentation axis is outside volume dimensions: {axis}")
        indices = presentation_spec.indices or [data.shape[axis] // 2]
        if any(index < 0 or index >= data.shape[axis] for index in indices):
            raise AssetAdapterError("Presentation slice index is outside the volume")
        if presentation_spec.axis is None or not presentation_spec.indices:
            raise AssetAdapterError("Volume presentation requires explicit axis and slice indices")
        policy = ({"encoding": "window", "low": presentation_spec.window[0], "high": presentation_spec.window[1],
                   "invert": loaded.metadata.get("invert", False)}
                  if presentation_spec.window else {"encoding": "preserve"})
        slices = [np.asarray(_render_pixels(policy=policy,
                  decoded=(np.take(data, index, axis=axis), asset_spec.format, None,
                           loaded.metadata.get("invert", False)))[0]) for index in indices]
        if presentation_spec.crop is not None:
            if len(presentation_spec.crop) != 4:
                raise AssetAdapterError("Presentation crop requires normalized xyxy coordinates")
            x0, y0, x1, y1 = presentation_spec.crop
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise AssetAdapterError("Presentation crop must be normalized xyxy")
            cropped = []
            for slice_data in slices:
                height, width = np.asarray(slice_data).shape[-2:]
                left, right = int(x0 * width), int(x1 * width)
                top, bottom = int(y0 * height), int(y1 * height)
                if right <= left or bottom <= top:
                    raise AssetAdapterError("Presentation crop produced an empty slice")
                cropped.append(np.asarray(slice_data)[top:bottom, left:right])
            slices = cropped
        if presentation_spec.resize is not None:
            from PIL import Image
            width, height = presentation_spec.resize
            slices = [np.asarray(Image.fromarray(slice).resize((width, height), Image.Resampling.NEAREST)) for slice in slices]
        if presentation_spec.mode == "slices":
            return LoadedAsset(np.stack(slices), {**loaded.metadata, "presentation": presentation_spec.model_dump()})
        return LoadedAsset(np.concatenate(slices, axis=1), {**loaded.metadata, "presentation": presentation_spec.model_dump()})

class ImageAdapter(AssetAdapter):
    kind = "image"
    formats = frozenset({"png", "jpeg", "jpg", "tiff", "bmp", "webp", "gif"})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        from PIL import Image
        path = _asset_path(asset_spec, source_root)
        with Image.open(path) as image:
            return {"kind": "image", "format": image.format, "width": image.width, "height": image.height}

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        from .readers import RasterReader
        pixels, _, _, _ = RasterReader().read_pixels(_asset_path(asset_spec, source_root))
        return LoadedAsset(pixels, self.inspect(asset_spec, source_root=source_root))


class DicomImageAdapter(AssetAdapter):
    kind = "image"
    formats = frozenset({"dicom"})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        return self.load(asset_spec, source_root=source_root).metadata

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        from .readers import DicomReader
        pixels, _, window, inverse = DicomReader().read_pixels(_asset_path(asset_spec, source_root))
        return LoadedAsset(pixels, {"window": window, "invert": inverse})


class NiftiVolumeAdapter(AssetAdapter):
    kind = "volume"
    formats = frozenset({"nifti"})

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise AssetAdapterError("nibabel is required for NIfTI assets") from exc
        image = nib.load(str(_asset_path(asset_spec, source_root)))
        data = np.asanyarray(image.dataobj)
        if data.ndim < 3:
            raise AssetAdapterError("NIfTI asset is not a volume")
        return LoadedAsset(data, {"shape": list(image.shape), "dtype": str(data.dtype), "spacing": list(image.header.get_zooms()),
                                  "origin": image.affine[:3, 3].tolist(), "direction": None, "affine": image.affine.tolist(),
                                  "axis_order": list(range(data.ndim)), "slice_count": int(data.shape[2]),
                                  "frame_count": int(data.shape[3]) if data.ndim > 3 else 1})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        return self.load(asset_spec, source_root=source_root).metadata


class NrrdVolumeAdapter(AssetAdapter):
    kind = "volume"
    formats = frozenset({"nrrd", "itk"})

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        path = _asset_path(asset_spec, source_root)
        if path.suffix.lower() in {".nrrd", ".nhdr"}:
            try:
                import nrrd
            except ImportError as exc:
                raise AssetAdapterError("pynrrd is required for NRRD assets") from exc
            array, header = nrrd.read(str(path), index_order="F")
            directions = header.get("space directions")
            metadata = {"spacing": np.linalg.norm(directions, axis=1).tolist() if directions is not None else list(header.get("spacings", [])),
                        "origin": header.get("space origin"), "direction": directions}
        else:
            try:
                import SimpleITK as sitk
            except ImportError as exc:
                raise AssetAdapterError("SimpleITK is required for ITK assets") from exc
            image = sitk.ReadImage(str(path))
            array = np.asarray(sitk.GetArrayFromImage(image)).transpose()
            metadata = {"spacing": list(image.GetSpacing()), "origin": list(image.GetOrigin()),
                        "direction": np.asarray(image.GetDirection()).reshape(image.GetDimension(), -1).T.tolist()}
        if array.ndim != 3:
            raise AssetAdapterError("Expected a three-dimensional volume")
        return LoadedAsset(array, {**metadata, "shape": list(array.shape), "dtype": str(array.dtype),
                                  "axis_order": [0, 1, 2], "slice_count": int(array.shape[2])})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        return self.load(asset_spec, source_root=source_root).metadata


class DicomSeriesAdapter(AssetAdapter):
    kind = "volume"
    formats = frozenset({"dicom_series"})

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        try:
            import pydicom
        except ImportError as exc:
            raise AssetAdapterError("pydicom is required for DICOM assets") from exc
        root = _asset_path(asset_spec, source_root)
        members = asset_spec.metadata.get("members")
        if members:
            base = Path(source_root).resolve() if source_root is not None else Path.cwd()
            files = [(base / item).resolve() for item in members]
        else:
            files = sorted(p for p in root.rglob("*") if p.is_file())
        try:
            datasets = [pydicom.dcmread(str(path)) for path in files]
        except Exception as exc:
            raise AssetAdapterError(f"DICOM Series cannot be decoded: {exc}") from exc
        if not datasets or any(not hasattr(ds, "PixelData") for ds in datasets):
            raise AssetAdapterError("DICOM Series has missing Pixel Data")
        uids = {str(getattr(ds, "SeriesInstanceUID", "")) for ds in datasets}
        if len(uids) != 1:
            raise AssetAdapterError("DICOM Series contains multiple SeriesInstanceUID values")
        studies = {str(getattr(ds, "StudyInstanceUID", "")) for ds in datasets}
        patients = {str(getattr(ds, "PatientID", "")) for ds in datasets}
        if len(studies) != 1 or len(patients) > 1:
            raise AssetAdapterError("DICOM Series contains mixed StudyInstanceUID or PatientID values")
        orientations = {tuple(float(x) for x in getattr(ds, "ImageOrientationPatient", [])) for ds in datasets}
        if len(orientations) > 1:
            raise AssetAdapterError("DICOM Series has inconsistent orientation")
        orientation = next(iter(orientations), ())
        normal = np.cross(np.asarray(orientation[:3], dtype=float), np.asarray(orientation[3:], dtype=float))
        if np.linalg.norm(normal) == 0:
            raise AssetAdapterError("DICOM Series has invalid orientation")
        normal = normal / np.linalg.norm(normal)
        def key(ds):
            position = getattr(ds, "ImagePositionPatient", None)
            projection = float(np.dot(normal, np.asarray(position, dtype=float))) if position is not None else float("inf")
            return (projection, int(getattr(ds, "InstanceNumber", 0)), str(getattr(ds, "SOPInstanceUID", "")))
        datasets.sort(key=key)
        seen = [str(getattr(ds, "SOPInstanceUID", "")) for ds in datasets]
        if len(seen) != len(set(seen)):
            raise AssetAdapterError("DICOM Series contains duplicate SOPInstanceUID values")
        arrays = []
        for ds in datasets:
            if int(getattr(ds, "NumberOfFrames", 1)) != 1:
                raise AssetAdapterError("Multiframe DICOM must be represented as a separate volume Asset")
            if str(getattr(ds, "PhotometricInterpretation", "")) not in {"MONOCHROME1", "MONOCHROME2"}:
                raise AssetAdapterError("Unsupported DICOM Series photometric interpretation")
            from .readers import DicomReader
            pixels, _, _, _ = DicomReader().read_pixels(ds.filename)
            arrays.append(pixels)
        shapes = {tuple(array.shape) for array in arrays}
        if len(shapes) != 1:
            raise AssetAdapterError("DICOM Series has inconsistent geometry")
        positions = [getattr(ds, "ImagePositionPatient", None) for ds in datasets]
        if all(position is not None for position in positions) and len(positions) >= 2:
            distances = [float(np.dot(normal, np.asarray(positions[i + 1], dtype=float) - np.asarray(positions[i], dtype=float))) for i in range(len(positions) - 1)]
            if any(abs(distance) < 1e-9 for distance in distances):
                raise AssetAdapterError("DICOM Series contains duplicate slice positions")
            step = distances[0]
            if any(abs(distance - step) > max(abs(step) * 0.05, 1e-6) for distance in distances[1:]):
                raise AssetAdapterError("DICOM Series has a missing or irregular slice gap")
        volume = np.stack(arrays)
        first = datasets[0]
        return LoadedAsset(volume, {"shape": list(volume.shape), "dtype": str(volume.dtype),
                                    "spacing": list(getattr(first, "PixelSpacing", [])),
                                    "origin": list(getattr(first, "ImagePositionPatient", [])),
                                    "direction": list(getattr(first, "ImageOrientationPatient", [])),
                                    "affine": None, "axis_order": [0, 1, 2],
                                    "slice_count": len(datasets), "frame_count": 1,
                                    "invert": str(getattr(first, "PhotometricInterpretation", "")) == "MONOCHROME1",
                                    "series_uid": next(iter(uids))})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        return self.load(asset_spec, source_root=source_root).metadata


ADAPTERS = (DicomImageAdapter(), ImageAdapter(), NiftiVolumeAdapter(), NrrdVolumeAdapter(), DicomSeriesAdapter())


def adapter_for(asset: AssetEntry) -> AssetAdapter:
    for adapter in ADAPTERS:
        if adapter.kind == asset.kind and asset.format.lower() in adapter.formats:
            return adapter
    raise AssetAdapterError(f"No adapter for {asset.kind}/{asset.format}")


def _asset_path(asset: AssetEntry, source_root: str | Path | None) -> Path:
    path = Path(asset.source_path)
    if path.is_absolute():
        raise AssetAdapterError("Asset source_path must be source-relative")
    return (Path(source_root).resolve() / path).resolve() if source_root is not None else path



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



def _pixel_info(pixels, detected, window, inverse):
    import numpy as np
    from PIL import Image
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



def _render_pixels(*, decoded, policy):
    """Preserve representable pixels or apply an explicitly selected linear display window."""
    import numpy as np
    from PIL import Image

    policy = policy or {"encoding": "preserve"}
    pixels, detected, window, inverse = decoded
    info = _pixel_info(pixels, detected, window, inverse)
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




SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}

def verify_image(path, expected_digest=None):
    path = Path(path)
    if expected_digest and digest_file(path) != expected_digest:
        raise ValueError(f"Presentation digest mismatch: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format not in SUPPORTED_IMAGE_FORMATS or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Unsupported model image: {path}")
        return {"format": image.format, "width": image.width, "height": image.height}


def materialize_image(asset, spec, destination, *, source_root):
    """Render an explicitly requested derived image as PNG; direct images never call this."""
    destination = Path(destination)
    if destination.suffix.lower() != ".png":
        raise ValueError("Converted images require a .png target")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = adapter_for(asset).present(asset, spec, source_root=source_root).data
    if spec.mode == "slices":
        if np.asarray(pixels).shape[0] != 1:
            raise ValueError("Select one slice per PNG, or use an explicit montage")
        pixels = pixels[0]
    image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(np.asarray(pixels))
    image.save(destination, format="PNG")
    return {**verify_image(destination), "output_digest": digest_file(destination)}
