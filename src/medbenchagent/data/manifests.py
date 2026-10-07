"""Canonical presentation, split and sampling manifest contracts.

These manifests are prepared-data inputs.  They freeze the visual material
shown to a model and the order in which related evidence is consumed.  The
benchmark builder only validates and references them; it does not infer an
order from filenames or Case dictionary order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medbenchagent.data.sources import digest_file


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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


def iter_image_manifest(path):
    from .models import ImageEntry
    seen = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image = ImageEntry.model_validate_json(line)
        if image.path.casefold() in seen:
            raise ValueError("Duplicate image path")
        seen.add(image.path.casefold())
        yield image


def write_image_manifest(path, entries):
    Path(path).write_text("".join(row.model_dump_json() + "\n" for row in entries), encoding="utf-8")


def dataset_images(cases_path, manifest_path):
    """Resolve converted records and derive ordinary source access without a second manifest."""
    from .models import CaseRecord, ImageEntry
    from .asset_adapters import verify_image
    import hashlib
    root = Path(cases_path).parent
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    images = {row.path: row for row in iter_image_manifest(manifest_path)}
    needed = {e.value for line in Path(cases_path).read_text(encoding="utf-8").splitlines() if line.strip()
              for e in CaseRecord.model_validate_json(line).evidence if e.type == "asset" and e.mask is None}
    if not set(images) <= needed:
        raise ValueError("Converted manifest contains unreferenced images")
    direct = {}
    for locator in sorted(needed - images.keys()):
        path = Path(locator)
        if not path.is_absolute():
            raise ValueError("Direct image must reference its original absolute source path")
        sources = [(key, Path(value).resolve()) for key, value in metadata.get("source_roots", {}).items()
                   if path.resolve().is_relative_to(Path(value).resolve())]
        if len(sources) != 1:
            raise ValueError("Direct image must belong to one frozen source")
        source_id, source_root = sources[0]
        info = verify_image(path)
        digest = digest_file(path)
        direct[locator] = digest
        images[locator] = ImageEntry(path=locator, file_digest=digest, format=info["format"].lower(),
            width=info["width"], height=info["height"], source_id=source_id,
            source_path=path.resolve().relative_to(source_root).as_posix(), source_digest=digest)
    if "direct_images_digest" in metadata:
        actual = hashlib.sha256(json.dumps(direct, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if actual != metadata["direct_images_digest"]:
            raise ValueError("Direct image source content changed")
        for image in images.values():
            verify_image(root / image.path, image.file_digest)
    elif direct:
        raise ValueError("Direct image content binding is missing")
    return images
