"""Canonical annotation data contracts."""

from __future__ import annotations

from typing import Any, Literal
import json
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceRef(CanonicalModel):
    """A source location, not an executable transformation expression."""

    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    record: str | None = None
    field: str | None = None


class EntityRef(CanonicalModel):
    """Dataset-scoped clinical identity without a separate entity database."""

    kind: str = Field(min_length=1)
    id: str = Field(min_length=1)


class EvidenceRef(CanonicalModel):
    case_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)


class CaseRelation(CanonicalModel):
    subject: EvidenceRef | EntityRef
    predicate: str = Field(min_length=1)
    object: EvidenceRef | EntityRef
    source_refs: list[SourceRef] = Field(min_length=1)


class MaskPlaneFeature(CanonicalModel):
    """Deterministic geometry for one indexed plane of a mask."""

    axis: int = Field(ge=0)
    index: int = Field(ge=0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    pixel_count: int = Field(gt=0)
    area_fraction: float = Field(ge=0, le=1)
    centroid: list[float] = Field(min_length=2, max_length=2)
    equivalent_diameter_pixels: float = Field(gt=0)

    @model_validator(mode="after")
    def geometry(self):
        if not (0 <= self.bbox[0] < self.bbox[2] <= 1 and 0 <= self.bbox[1] < self.bbox[3] <= 1):
            raise ValueError("Mask plane bbox requires normalized xyxy coordinates")

        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.centroid):
            raise ValueError("Mask plane centroid requires normalized finite coordinates")

        if not math.isfinite(self.area_fraction) or not math.isfinite(
            self.equivalent_diameter_pixels
        ):
            raise ValueError("Mask plane geometry must be finite")
        expected = 2 * math.sqrt(self.pixel_count / math.pi)
        if not math.isclose(self.equivalent_diameter_pixels, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Mask plane equivalent_diameter_pixels does not match pixel_count")
        return self


class MaskFeatureSet(CanonicalModel):
    """Prepared, source-digest-bound mask geometry.

    The host consumes these values as immutable derived evidence.  It never
    re-reads a mask file while planning or generating a benchmark item.
    """

    source_digest: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    channel: Literal["red", "green", "blue", "alpha", "luminance", "any"]
    foreground: list[int] = Field(min_length=1)
    shape: list[int] = Field(min_length=2, max_length=3)
    bbox: list[float] = Field(min_length=4, max_length=4)
    pixel_count: int = Field(gt=0)
    area_fraction: float = Field(ge=0, le=1)
    centroid: list[float] = Field(min_length=2, max_length=3)
    component_count: int = Field(gt=0)
    equivalent_diameter_pixels: float = Field(gt=0)
    spacing: list[float] | None = None
    physical_area: float | None = Field(default=None, gt=0)
    physical_unit: str | None = None
    planes: list[MaskPlaneFeature] = Field(default_factory=list)

    @model_validator(mode="after")
    def geometry(self):
        if any(type(value) is not int or value < 0 or value > 255 for value in self.foreground):
            raise ValueError("Mask foreground values must be integers in [0,255]")

        if len(set(self.foreground)) != len(self.foreground):
            raise ValueError("Mask foreground values must be unique")

        if any(type(value) is not int or value <= 0 for value in self.shape):
            raise ValueError("Mask shape dimensions must be positive integers")

        if not (0 <= self.bbox[0] < self.bbox[2] <= 1 and 0 <= self.bbox[1] < self.bbox[3] <= 1):
            raise ValueError("Mask bbox requires normalized xyxy coordinates")

        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.centroid):
            raise ValueError("Mask centroid requires normalized finite coordinates")

        if not math.isfinite(self.area_fraction) or not math.isfinite(
            self.equivalent_diameter_pixels
        ):
            raise ValueError("Mask geometry must be finite")
        expected = (
            2 * (3 * self.pixel_count / (4 * math.pi)) ** (1 / 3)
            if len(self.shape) == 3
            else 2 * math.sqrt(self.pixel_count / math.pi)
        )
        if not math.isclose(self.equivalent_diameter_pixels, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Mask equivalent_diameter_pixels does not match pixel_count")
        if self.spacing is not None:
            if len(self.spacing) not in {2, 3} or any(
                not math.isfinite(value) or value <= 0 for value in self.spacing
            ):
                raise ValueError("Mask spacing must contain positive finite values")
        if self.physical_area is not None:
            if self.spacing is None or not self.physical_unit or not self.physical_unit.strip():
                raise ValueError("physical_area requires spacing and physical_unit")

            if not math.isfinite(self.physical_area):
                raise ValueError("physical_area must be finite")
        elif self.physical_unit is not None and not self.physical_unit.strip():
            raise ValueError("physical_unit must be non-empty when supplied")
        coordinates = {(plane.axis, plane.index) for plane in self.planes}
        if len(coordinates) != len(self.planes):
            raise ValueError("Mask planes must have unique (axis,index) pairs")

        if any(plane.axis >= len(self.shape) for plane in self.planes):
            raise ValueError("Mask plane axis is outside mask shape")

        if len(self.shape) == 3 and any(plane.axis != 2 for plane in self.planes):
            raise ValueError("Volume mask planes must use axis 2")

        if any(plane.index >= self.shape[plane.axis] for plane in self.planes):
            raise ValueError("Mask plane index is outside mask shape")
        return self


class Evidence(CanonicalModel):
    id: str = Field(min_length=1)
    type: Literal["value", "text", "asset", "region"]
    name: str = Field(min_length=1)
    value: Any
    time: float | int | None = None
    unit: str | None = None
    ref: list[str] = Field(default_factory=list)
    mask: MaskFeatureSet | None = None
    view_id: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    input_visibility: Literal["model_visible", "reference_only", "unknown"] = "unknown"
    entity: EntityRef | None = None
    study_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def value_contract(self):
        json.dumps(self.model_dump(), allow_nan=False)
        if self.type == "value" and (
            isinstance(self.value, dict)
            or isinstance(self.value, list)
            and any(isinstance(v, (list, dict)) for v in self.value)
        ):
            raise ValueError("value Evidence requires a scalar or flat list")

        if self.type == "text" and not isinstance(self.value, str):
            raise ValueError("text Evidence requires a string value")

        if self.type == "asset" and (not isinstance(self.value, str) or not self.value):
            raise ValueError("asset Evidence requires a stable Asset ID")

        if self.mask is not None and self.type != "asset":
            raise ValueError("Mask features are only valid on asset Evidence")

        if self.mask is not None and self.input_visibility == "model_visible":
            raise ValueError("Mask features cannot be model-visible input")

        if self.view_id is not None and not self.view_id.strip():
            raise ValueError("view_id must be a non-empty stable identifier")
        if self.type == "region":
            if (
                not isinstance(self.value, list)
                or len(self.value) != 4
                or any(type(v) not in (int, float) for v in self.value)
                or not 0 <= self.value[0] < self.value[2] <= 1
                or not 0 <= self.value[1] < self.value[3] <= 1
            ):
                raise ValueError("region Evidence requires normalized xyxy coordinates")
        if self.unit is not None and (self.type != "value" or not self.unit.strip()):
            raise ValueError("unit is only valid for value Evidence")

        if any(not isinstance(ref, str) or not ref for ref in self.ref):
            raise ValueError("Evidence refs must be nonempty strings")
        return self


class CaseRecord(CanonicalModel):
    case_id: str = Field(min_length=1)
    patient_id: str | None = None
    evidence: list[Evidence]
    relations: list[CaseRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relations(self):
        return validate_case(self)


class AssetEntry(CanonicalModel):
    asset_id: str = Field(min_length=1)
    kind: Literal["image", "volume"]
    format: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    file_digest: str = Field(min_length=1)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_id: str = Field(default="source", min_length=1)

    @model_validator(mode="after")
    def semantic_metadata(self):
        role = self.metadata.get("semantic_role")
        visibility = self.metadata.get("input_visibility")
        if role is not None and role not in {
            "primary",
            "model_input",
            "image",
            "volume",
            "annotation",
            "mask",
            "label",
            "calibration",
            "reference",
        }:
            raise ValueError("Asset semantic_role is not supported")

        if visibility is not None and visibility not in {
            "model_visible",
            "reference_only",
            "unknown",
        }:
            raise ValueError("Asset input_visibility is not supported")
        return self


class PresentationSpec(CanonicalModel):
    mode: Literal["native", "slices", "montage"] = "native"
    axis: int | None = Field(default=None, ge=0, le=2)
    indices: list[int] = Field(default_factory=list)
    window: tuple[float, float] | None = None
    resize: tuple[int, int] | None = None
    crop: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def geometry(self):
        json.dumps(self.model_dump(), allow_nan=False)
        if any(i < 0 for i in self.indices) or len(self.indices) != len(set(self.indices)):
            raise ValueError("Slice indices must be unique nonnegative integers")

        if self.window and self.window[0] >= self.window[1]:
            raise ValueError("Window requires low < high")

        if self.resize and any(v <= 0 for v in self.resize):
            raise ValueError("Resize dimensions must be positive")

        if self.crop and not (
            0 <= self.crop[0] < self.crop[2] <= 1 and 0 <= self.crop[1] < self.crop[3] <= 1
        ):
            raise ValueError("Crop requires normalized xyxy")
        return self


def validate_case(case: CaseRecord, assets: dict[str, AssetEntry] | None = None) -> CaseRecord:
    ids = {item.id for item in case.evidence}
    if len(ids) != len(case.evidence):
        raise ValueError(f"Duplicate Evidence ID in Case {case.case_id}")
    for item in case.evidence:
        if any(ref not in ids for ref in item.ref):
            raise ValueError(f"Evidence {item.id} references an unknown Evidence")

        if (
            item.type == "asset"
            and item.mask is None
            and assets is not None
            and item.value not in assets
        ):
            raise ValueError(f"Evidence {item.id} references an unknown Asset ID")

        if item.id in item.ref or len(item.ref) != len(set(item.ref)):
            raise ValueError("Self or duplicate Evidence ref")
        if item.type == "region":
            linked = [value for value in case.evidence if value.id in item.ref]
            if len(linked) != 1 or linked[0].type != "asset":
                raise ValueError("Region must reference exactly one Asset Evidence")

            if assets and linked[0].value not in assets:
                raise ValueError(f"Region {item.id} references an unknown Asset")
    graph = {item.id: item.ref for item in case.evidence}

    def visit(identifier, path):
        if identifier in path:
            raise ValueError("Cyclic Evidence relation")
        for parent in graph[identifier]:
            visit(parent, path | {identifier})

    for identifier in graph:
        visit(identifier, set())
    for relation in case.relations:
        for endpoint in (relation.subject, relation.object):
            if isinstance(endpoint, EvidenceRef):
                if endpoint.case_id != case.case_id or endpoint.evidence_id not in ids:
                    raise ValueError("Case relation references an unknown local Evidence")
    return case
