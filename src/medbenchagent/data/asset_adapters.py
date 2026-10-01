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

    def inspect(self, asset_spec: AssetEntry) -> dict[str, Any]:
        raise NotImplementedError

    def validate(self, asset_spec: AssetEntry) -> list[str]:
        try:
            self.inspect(asset_spec)
        except Exception as exc:
            return [str(exc)]
        return []

    def load(self, asset_spec: AssetEntry) -> LoadedAsset:
        raise NotImplementedError

    def present(self, asset_spec: AssetEntry, presentation_spec: PresentationSpec) -> LoadedAsset:
        loaded = self.load(asset_spec)
        if presentation_spec.mode == "native":
            return loaded
        data = np.asarray(loaded.data)
        if data.ndim < 3:
            raise AssetAdapterError("slice presentation requires a volume")
        axis = presentation_spec.axis if presentation_spec.axis is not None else 0
        indices = presentation_spec.indices or [data.shape[axis] // 2]
        slices = [np.take(data, index, axis=axis) for index in indices]
        if presentation_spec.mode == "slices":
            return LoadedAsset(np.stack(slices), {**loaded.metadata, "presentation": presentation_spec.model_dump()})
        return LoadedAsset(np.concatenate(slices, axis=1), {**loaded.metadata, "presentation": presentation_spec.model_dump()})


class ImageAdapter(AssetAdapter):
    kind = "image"
    formats = frozenset({"png", "jpeg", "jpg", "dicom"})

    def inspect(self, asset_spec: AssetEntry) -> dict[str, Any]:
        from PIL import Image
        path = Path(asset_spec.source_path)
        with Image.open(path) as image:
            return {"kind": "image", "format": image.format, "width": image.width, "height": image.height}

    def load(self, asset_spec: AssetEntry) -> LoadedAsset:
        from PIL import Image
        with Image.open(asset_spec.source_path) as image:
            image.load()
            return LoadedAsset(image.copy(), self.inspect(asset_spec))


class NiftiVolumeAdapter(AssetAdapter):
    kind = "volume"
    formats = frozenset({"nifti"})

    def load(self, asset_spec: AssetEntry) -> LoadedAsset:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise AssetAdapterError("nibabel is required for NIfTI assets") from exc
        image = nib.load(asset_spec.source_path)
        data = np.asanyarray(image.dataobj)
        if data.ndim < 3:
            raise AssetAdapterError("NIfTI asset is not a volume")
        return LoadedAsset(data, {"shape": list(image.shape), "dtype": str(data.dtype), "affine": image.affine.tolist()})

    def inspect(self, asset_spec: AssetEntry) -> dict[str, Any]:
        return self.load(asset_spec).metadata


class DicomSeriesAdapter(AssetAdapter):
    kind = "volume"
    formats = frozenset({"dicom_series"})

    def load(self, asset_spec: AssetEntry) -> LoadedAsset:
        try:
            import pydicom
        except ImportError as exc:
            raise AssetAdapterError("pydicom is required for DICOM assets") from exc
        root = Path(asset_spec.source_path)
        files = sorted(p for p in root.rglob("*") if p.is_file())
        datasets = [pydicom.dcmread(str(path)) for path in files]
        if not datasets or any(not hasattr(ds, "PixelData") for ds in datasets):
            raise AssetAdapterError("DICOM Series has missing Pixel Data")
        uids = {str(getattr(ds, "SeriesInstanceUID", "")) for ds in datasets}
        if len(uids) != 1:
            raise AssetAdapterError("DICOM Series contains multiple SeriesInstanceUID values")
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
        volume = np.stack(arrays)
        first = datasets[0]
        return LoadedAsset(volume, {"shape": list(volume.shape), "dtype": str(volume.dtype),
                                    "spacing": list(getattr(first, "PixelSpacing", [])),
                                    "series_uid": next(iter(uids))})

    def inspect(self, asset_spec: AssetEntry) -> dict[str, Any]:
        return self.load(asset_spec).metadata


ADAPTERS = (ImageAdapter(), NiftiVolumeAdapter(), DicomSeriesAdapter())


def adapter_for(asset: AssetEntry) -> AssetAdapter:
    for adapter in ADAPTERS:
        if adapter.kind == asset.kind and asset.format.lower() in adapter.formats:
            return adapter
    raise AssetAdapterError(f"No adapter for {asset.kind}/{asset.format}")
