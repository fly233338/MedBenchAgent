"""Canonical annotation data contracts."""

from __future__ import annotations

from typing import Any, Literal
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(CanonicalModel):
    id: str = Field(min_length=1)
    type: Literal["value", "text", "asset", "region"]
    name: str = Field(min_length=1)
    value: Any
    time: float | int | None = None
    unit: str | None = None
    ref: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def value_contract(self):
        json.dumps(self.model_dump(), allow_nan=False)
        if self.type == "value" and (isinstance(self.value, dict) or isinstance(self.value, list) and any(isinstance(v, (list, dict)) for v in self.value)):
            raise ValueError("value Evidence requires a scalar or flat list")
        if self.type == "text" and not isinstance(self.value, str):
            raise ValueError("text Evidence requires a string value")
        if self.type == "asset" and (not isinstance(self.value, str) or not self.value):
            raise ValueError("asset Evidence requires an asset_id string")
        if self.type == "region":
            if (not isinstance(self.value, list) or len(self.value) != 4 or
                    any(type(v) not in (int, float) for v in self.value) or
                    not 0 <= self.value[0] < self.value[2] <= 1 or
                    not 0 <= self.value[1] < self.value[3] <= 1):
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

    @model_validator(mode="after")
    def relations(self):
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
        if self.crop and not (0 <= self.crop[0] < self.crop[2] <= 1 and 0 <= self.crop[1] < self.crop[3] <= 1):
            raise ValueError("Crop requires normalized xyxy")
        return self


class EvidenceProfile(CanonicalModel):
    name: str
    cases: int = 0
    types: dict[str, int] = Field(default_factory=dict)
    value_types: dict[str, int] = Field(default_factory=dict)
    units: dict[str, int] = Field(default_factory=dict)
    has_time: int = 0
    has_ref: int = 0


class AssetProfile(CanonicalModel):
    kinds: dict[str, int] = Field(default_factory=dict)
    formats: dict[str, int] = Field(default_factory=dict)
    assets_per_case: dict[int, int] = Field(default_factory=dict)
    volume_properties: dict[str, Any] = Field(default_factory=dict)


class DatasetProfile(CanonicalModel):
    case_count: int = 0
    patient_count: int = 0
    cases_per_patient: dict[str, int] = Field(default_factory=dict)
    evidence: list[EvidenceProfile] = Field(default_factory=list)
    assets: AssetProfile = Field(default_factory=AssetProfile)


def validate_case(case: CaseRecord, assets: dict[str, AssetEntry] | None = None) -> CaseRecord:
    ids = {item.id for item in case.evidence}
    if len(ids) != len(case.evidence):
        raise ValueError(f"Duplicate Evidence ID in Case {case.case_id}")
    for item in case.evidence:
        if any(ref not in ids for ref in item.ref):
            raise ValueError(f"Evidence {item.id} references an unknown Evidence")
        if item.type == "asset" and assets is not None and item.value not in assets:
            raise ValueError(f"Evidence {item.id} references an unknown Asset")
        if item.id in item.ref or len(item.ref) != len(set(item.ref)):
            raise ValueError("Self or duplicate Evidence ref")
        if item.type == "region":
            linked = [value for value in case.evidence if value.id in item.ref]
            if len(linked) != 1 or linked[0].type != "asset":
                raise ValueError("Region must reference exactly one Asset Evidence")
            if assets and assets[linked[0].value].kind != "image":
                raise ValueError("2D Region requires an image Asset")
    graph = {item.id: item.ref for item in case.evidence}
    def visit(identifier, path):
        if identifier in path:
            raise ValueError("Cyclic Evidence relation")
        for parent in graph[identifier]:
            visit(parent, path | {identifier})
    for identifier in graph:
        visit(identifier, set())
    return case

