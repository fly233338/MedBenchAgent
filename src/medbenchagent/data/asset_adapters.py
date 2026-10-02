"""Native Asset readers and deterministic presentation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medbenchagent.data.v5_models import AssetEntry, PresentationSpec


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
            return loaded
        data = np.asarray(loaded.data)
        if data.ndim < 3:
            raise AssetAdapterError("slice presentation requires a volume")
        axis = presentation_spec.axis if presentation_spec.axis is not None else 0
        indices = presentation_spec.indices or [data.shape[axis] // 2]
        if presentation_spec.window is not None:
            low, high = presentation_spec.window
            if low >= high:
                raise AssetAdapterError("Presentation window requires low < high")
            data = np.clip((data.astype(float) - low) / (high - low), 0, 1) * 255
            data = data.astype(np.uint8)
        if presentation_spec.crop is not None:
            if len(presentation_spec.crop) != 4:
                raise AssetAdapterError("Presentation crop requires normalized xyxy coordinates")
            y0, x0, y1, x1 = presentation_spec.crop
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise AssetAdapterError("Presentation crop must be normalized xyxy")
            h, w = data.shape[-2:]
            data = data[..., int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
        slices = [np.take(data, index, axis=axis) for index in indices]
        if presentation_spec.resize is not None:
            from PIL import Image
            width, height = presentation_spec.resize
            slices = [np.asarray(Image.fromarray(slice).resize((width, height))) for slice in slices]
        if presentation_spec.mode == "slices":
            return LoadedAsset(np.stack(slices), {**loaded.metadata, "presentation": presentation_spec.model_dump()})
        return LoadedAsset(np.concatenate(slices, axis=1), {**loaded.metadata, "presentation": presentation_spec.model_dump()})

    def presentation_digest(self, asset_spec: AssetEntry, presentation_spec: PresentationSpec) -> str:
        import hashlib
        import json
        payload = json.dumps({"asset": asset_spec.file_digest, "spec": presentation_spec.model_dump()},
                             sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def present_to_path(self, asset_spec: AssetEntry, presentation_spec: PresentationSpec, output: str | Path, *, source_root: str | Path | None = None) -> dict:
        from medbenchagent.data.sources import digest_file
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        loaded = self.present(asset_spec, presentation_spec, source_root=source_root)
        data = loaded.data
        try:
            from PIL import Image
            image = data if isinstance(data, Image.Image) else Image.fromarray(np.asarray(data))
            image.save(output, format="PNG", compress_level=9)
        except Exception as exc:
            raise AssetAdapterError(f"Cannot persist presentation: {exc}") from exc
        return {"asset_id": asset_spec.asset_id, "presentation_id": self.presentation_digest(asset_spec, presentation_spec),
                "mode": presentation_spec.mode, "axis": presentation_spec.axis, "indices": presentation_spec.indices,
                "window": presentation_spec.window, "resize": presentation_spec.resize, "crop": presentation_spec.crop,
                "adapter_version": "v5", "input_digest": asset_spec.file_digest, "output_digest": digest_file(output),
                "path": output.as_posix()}


class ImageAdapter(AssetAdapter):
    kind = "image"
    formats = frozenset({"png", "jpeg", "jpg", "dicom"})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        from PIL import Image
        path = _asset_path(asset_spec, source_root)
        with Image.open(path) as image:
            return {"kind": "image", "format": image.format, "width": image.width, "height": image.height}

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        from PIL import Image
        with Image.open(_asset_path(asset_spec, source_root)) as image:
            image.load()
            return LoadedAsset(image.copy(), self.inspect(asset_spec, source_root=source_root))


class DicomImageAdapter(AssetAdapter):
    kind = "image"
    formats = frozenset({"dicom"})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        return self.load(asset_spec, source_root=source_root).metadata

    def load(self, asset_spec: AssetEntry, *, source_root=None) -> LoadedAsset:
        try:
            import pydicom
        except ImportError as exc:
            raise AssetAdapterError("pydicom is required for DICOM assets") from exc
        dataset = pydicom.dcmread(str(_asset_path(asset_spec, source_root)))
        if not hasattr(dataset, "PixelData"):
            raise AssetAdapterError("DICOM has no Pixel Data")
        if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
            raise AssetAdapterError("Multiframe DICOM requires a volume adapter")
        if str(getattr(dataset, "PhotometricInterpretation", "")) not in {"MONOCHROME1", "MONOCHROME2"}:
            raise AssetAdapterError("Unsupported DICOM photometric interpretation")
        pixels = dataset.pixel_array
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = np.max(pixels) - pixels
        return LoadedAsset(pixels, {"rows": int(dataset.Rows), "columns": int(dataset.Columns),
                                    "series_uid": str(getattr(dataset, "SeriesInstanceUID", "")),
                                    "sop_uid": str(getattr(dataset, "SOPInstanceUID", ""))})


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
                                  "origin": [0.0, 0.0, 0.0], "direction": None, "affine": image.affine.tolist(),
                                  "axis_order": list(range(data.ndim)), "slice_count": int(data.shape[2]),
                                  "frame_count": int(data.shape[3]) if data.ndim > 3 else 1})

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
        datasets = [pydicom.dcmread(str(path)) for path in files]
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
        def key(ds):
            position = getattr(ds, "ImagePositionPatient", None)
            geometry = tuple(float(x) for x in position) if position is not None else ()
            return (geometry, int(getattr(ds, "InstanceNumber", 0)), str(getattr(ds, "SOPInstanceUID", "")))
        datasets.sort(key=key)
        seen = [str(getattr(ds, "SOPInstanceUID", "")) for ds in datasets]
        if len(seen) != len(set(seen)):
            raise AssetAdapterError("DICOM Series contains duplicate SOPInstanceUID values")
        arrays = [ds.pixel_array for ds in datasets]
        shapes = {tuple(array.shape) for array in arrays}
        if len(shapes) != 1:
            raise AssetAdapterError("DICOM Series has inconsistent geometry")
        positions = [getattr(ds, "ImagePositionPatient", None) for ds in datasets]
        if all(position is not None for position in positions) and len(positions) > 2:
            distances = [float(positions[i + 1][2]) - float(positions[i][2]) for i in range(len(positions) - 1)]
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
                                    "series_uid": next(iter(uids))})

    def inspect(self, asset_spec: AssetEntry, *, source_root=None) -> dict[str, Any]:
        return self.load(asset_spec, source_root=source_root).metadata


ADAPTERS = (DicomImageAdapter(), ImageAdapter(), NiftiVolumeAdapter(), DicomSeriesAdapter())


def adapter_for(asset: AssetEntry) -> AssetAdapter:
    for adapter in ADAPTERS:
        if adapter.kind == asset.kind and asset.format.lower() in adapter.formats:
            return adapter
    raise AssetAdapterError(f"No adapter for {asset.kind}/{asset.format}")


def record_presentation(path: str | Path, record: dict[str, Any]) -> None:
    """Append a deterministic presentation record to a JSONL manifest."""
    import json
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _asset_path(asset: AssetEntry, source_root: str | Path | None) -> Path:
    path = Path(asset.source_path)
    if path.is_absolute():
        raise AssetAdapterError("Asset source_path must be source-relative")
    return (Path(source_root).resolve() / path).resolve() if source_root is not None else path
