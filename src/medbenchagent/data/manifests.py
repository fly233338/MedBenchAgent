"""Canonical presentation, split and sampling manifest contracts.

These manifests are prepared-data inputs.  They freeze the visual material
shown to a model and the order in which related evidence is consumed.  The
benchmark builder only validates and references them; it does not infer an
order from filenames or Case dictionary order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medbenchagent.data.sources import digest_file


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PresentationRecord(ManifestModel):
    schema_version: Literal["presentation-record"] = "presentation-record"
    presentation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    input_digest: str = Field(min_length=1)
    output_digest: str = Field(min_length=1)
    mode: Literal["native", "slices", "montage"]
    view_id: str | None = None
    panel_id: str | None = None
    axis: int | None = Field(default=None, ge=0)
    index: int | None = Field(default=None, ge=0)
    order: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_identity(self):
        if self.view_id is not None and not self.view_id.strip():
            raise ValueError("Presentation view_id must be non-empty when supplied")
        if self.panel_id is not None and not self.panel_id.strip():
            raise ValueError("Presentation panel_id must be non-empty when supplied")
        if self.mode == "slices" and self.index is None:
            raise ValueError("Slice PresentationRecord requires an index")
        return self


class SplitRecord(ManifestModel):
    sampling_unit_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    group_id: str = Field(min_length=1)


class SplitManifest(ManifestModel):
    schema_version: Literal["split-manifest"] = "split-manifest"
    records: list[SplitRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_records(self):
        units = [record.sampling_unit_id for record in self.records]
        if len(units) != len(set(units)):
            raise ValueError("Split manifest sampling_unit_id values must be unique")
        return self


class SamplingUnit(ManifestModel):
    sampling_unit_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    view_ids: list[str] = Field(default_factory=list)
    ordered_evidence_ids: list[str] = Field(default_factory=list)
    panel_id: str | None = None
    eligible: bool
    reason: str | None = None
    order: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self):
        if any(not value.strip() for value in self.view_ids):
            raise ValueError("Sampling view_ids must be non-empty")
        if len(self.view_ids) != len(set(self.view_ids)):
            raise ValueError("Sampling view_ids must be unique")
        if any(not value.strip() for value in self.ordered_evidence_ids):
            raise ValueError("Sampling ordered_evidence_ids must be non-empty")
        if len(self.ordered_evidence_ids) != len(set(self.ordered_evidence_ids)):
            raise ValueError("Sampling ordered_evidence_ids must be unique")
        if self.panel_id is not None and not self.panel_id.strip():
            raise ValueError("Sampling panel_id must be non-empty when supplied")
        if not self.eligible and not self.reason:
            raise ValueError("Ineligible SamplingUnit requires a reason")
        return self


class SamplingManifest(ManifestModel):
    schema_version: Literal["sampling-manifest"] = "sampling-manifest"
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    units: list[SamplingUnit] = Field(default_factory=list)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_units(self):
        identifiers = [unit.sampling_unit_id for unit in self.units]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Sampling manifest sampling_unit_id values must be unique")
        orders = [unit.order for unit in self.units]
        if len(orders) != len(set(orders)):
            raise ValueError("Sampling manifest order values must be unique")
        return self

    def eligible_units(self) -> list[SamplingUnit]:
        return [unit for unit in sorted(self.units, key=lambda item: item.order) if unit.eligible]


ManifestT = TypeVar("ManifestT", bound=BaseModel)


def load_manifest(path: str | Path, model: type[ManifestT]) -> ManifestT:
    """Load a JSON or JSONL manifest through its strict model."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Manifest is missing: {source}")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Manifest is empty: {source}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        return model.model_validate(payload)
    if not isinstance(payload, list):
        raise ValueError(f"Manifest must contain an object or JSONL records: {source}")
    if model is PresentationRecord:
        records = [PresentationRecord.model_validate(item) for item in payload]
        return records  # type: ignore[return-value]
    if model is SplitManifest:
        return model.model_validate({"records": payload})
    if model is SamplingManifest:
        return model.model_validate({"dataset_id": "dataset", "revision": "1", "units": payload})
    return model.model_validate(payload)


def manifest_digest(path: str | Path) -> str:
    return digest_file(path)


def verify_manifest_digest(path: str | Path, expected: str | None) -> str:
    if not expected:
        raise ValueError("Manifest digest is required")
    actual = manifest_digest(path)
    if actual != expected:
        raise ValueError(f"Manifest digest mismatch: {path}")
    return actual


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    return str(destination)
