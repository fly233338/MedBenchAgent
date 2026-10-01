"""Canonical annotation.v5 data contracts."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class V5Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(V5Model):
    id: str = Field(min_length=1)
    type: Literal["value", "text", "asset", "region"]
    name: str = Field(min_length=1)
    value: Any
    time: float | int | None = None
    unit: str | None = None
    ref: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def value_contract(self):
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
        if self.type != "value" and self.unit is not None and self.type not in {"text", "region"}:
            raise ValueError("unit is only valid for value Evidence")
        return self


class CaseRecord(V5Model):
    case_id: str = Field(min_length=1)
    patient_id: str | None = None
    evidence: list[Evidence]


class AssetEntry(V5Model):
    asset_id: str = Field(min_length=1)
    kind: Literal["image", "volume"]
    format: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    file_digest: str = Field(min_length=1)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PresentationSpec(V5Model):
    mode: Literal["native", "slices", "montage"] = "native"
    axis: int | None = Field(default=None, ge=0, le=2)
    indices: list[int] = Field(default_factory=list)
    window: tuple[float, float] | None = None
    resize: tuple[int, int] | None = None
    crop: tuple[float, float, float, float] | None = None


class EvidenceProfile(V5Model):
    name: str
    cases: int = 0
    types: dict[str, int] = Field(default_factory=dict)
    units: dict[str, int] = Field(default_factory=dict)
    has_time: int = 0
    has_ref: int = 0


class AssetProfile(V5Model):
    kinds: dict[str, int] = Field(default_factory=dict)
    formats: dict[str, int] = Field(default_factory=dict)
    assets_per_case: dict[int, int] = Field(default_factory=dict)


class DatasetProfile(V5Model):
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
    return case
