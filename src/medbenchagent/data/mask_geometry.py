"""Deterministic 2D/3D mask geometry computed by the host."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Iterable

import numpy as np

from medbenchagent.data.models import MaskFeatureSet, MaskPlaneFeature


ALGORITHM = "voxel-mask-geometry"


def _equivalent_diameter(count: int, dimensions: int) -> float:
    if dimensions == 3:
        return 2.0 * (3.0 * count / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 2.0 * math.sqrt(count / math.pi)


def _components(mask: np.ndarray, offsets: Iterable[tuple[int, ...]]) -> int:
    visited = np.zeros(mask.shape, dtype=bool)
    count = 0
    shape = mask.shape
    for origin in zip(*np.nonzero(mask)):
        if visited[origin]:
            continue
        count += 1
        queue = deque([origin])
        visited[origin] = True
        while queue:
            current = queue.popleft()
            for offset in offsets:
                neighbor = tuple(current[i] + offset[i] for i in range(mask.ndim))
                if any(index < 0 or index >= shape[i] for i, index in enumerate(neighbor)):
                    continue
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
    return count


def _bbox_centroid(
    mask: np.ndarray, *, x_axis: int, y_axis: int
) -> tuple[list[float], list[float]]:
    coordinates = np.nonzero(mask)
    x = coordinates[x_axis]
    y = coordinates[y_axis]
    width = mask.shape[x_axis]
    height = mask.shape[y_axis]
    bbox = [
        float(x.min() / width),
        float(y.min() / height),
        float((x.max() + 1) / width),
        float((y.max() + 1) / height),
    ]
    centroid = [
        float((x.astype(float) + 0.5).mean() / width),
        float((y.astype(float) + 0.5).mean() / height),
    ]
    return bbox, centroid


def _plane(mask: np.ndarray, index: int) -> MaskPlaneFeature:
    bbox, centroid = _bbox_centroid(mask, x_axis=0, y_axis=1)
    count = int(mask.sum())
    return MaskPlaneFeature(
        axis=2,
        index=index,
        bbox=bbox,
        pixel_count=count,
        area_fraction=float(count / mask.size),
        centroid=centroid,
        equivalent_diameter_pixels=_equivalent_diameter(count, 2),
    )


def extract_mask_features(
    data: np.ndarray,
    *,
    source_digest: str,
    foreground: list[int],
    channel: str = "any",
    spacing: list[float] | None = None,
    physical_unit: str | None = None,
) -> MaskFeatureSet:
    """Return canonical geometry without changing array axis order."""
    array = np.asarray(data)
    if array.ndim not in {2, 3}:
        raise ValueError("Mask geometry requires a 2D image or 3D volume")

    if not foreground:
        raise ValueError("Mask geometry requires at least one foreground value")
    mask = np.isin(array, np.asarray(foreground))
    if not bool(mask.any()):
        raise ValueError("Mask foreground is empty")
    if array.ndim == 3:
        bbox, centroid = _bbox_centroid(mask, x_axis=0, y_axis=1)
        offsets = [offset for offset in np.ndindex((3, 3, 3)) if offset != (1, 1, 1)]
        offsets = [tuple(value - 1 for value in offset) for offset in offsets]
        planes = [
            _plane(mask[:, :, index], index)
            for index in range(mask.shape[2])
            if bool(mask[:, :, index].any())
        ]
        component_count = _components(mask, offsets)
    else:
        bbox, centroid = _bbox_centroid(mask, x_axis=1, y_axis=0)
        offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        planes = []
        component_count = _components(mask, offsets)
    count = int(mask.sum())
    known_spacing = spacing if physical_unit and spacing else None
    physical_area = None
    if array.ndim == 2 and known_spacing and len(known_spacing) >= 2:
        physical_area = float(count * known_spacing[0] * known_spacing[1])
    return MaskFeatureSet(
        source_digest=source_digest,
        algorithm=ALGORITHM,
        channel=channel,
        foreground=list(foreground),
        shape=list(array.shape),
        bbox=bbox,
        pixel_count=count,
        area_fraction=float(count / mask.size),
        centroid=centroid,
        component_count=component_count,
        equivalent_diameter_pixels=_equivalent_diameter(count, array.ndim),
        spacing=known_spacing,
        physical_area=physical_area,
        physical_unit=physical_unit if known_spacing else None,
        planes=planes,
    )


def geometry_digest(features: MaskFeatureSet) -> str:
    payload = features.model_dump_json(exclude_none=True, by_alias=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
