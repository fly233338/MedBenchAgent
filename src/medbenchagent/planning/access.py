"""Planning-time DatasetAccessReceipt contract."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetAccessReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "dataset-access"
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    cases_path: str = Field(min_length=1)
    asset_manifest_path: str = Field(min_length=1)
    profile_path: str = Field(min_length=1)
    duckdb_path: str = Field(min_length=1)
    annotation_digest: str = Field(min_length=1)
    asset_manifest_digest: str = Field(min_length=1)
    profile_digest: str = Field(min_length=1)
    duckdb_digest: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    patient_count: int = Field(ge=0)
    patient_identity: str = Field(min_length=1)
    mapping_path: str | None = None
    mapping_digest: str | None = None
    dataset_card_path: str | None = None
    dataset_card_digest: str | None = None
    source_schema_version: str | None = None
    source_access: dict[str, Any] | None = None
    source_root: str | None = None
    source_roots: dict[str, str] = Field(default_factory=dict)
    bundle_root: str | None = None
    presentation_manifest_path: str | None = Field(default=None, min_length=1)
    presentation_manifest_digest: str | None = Field(default=None, min_length=1)
    split_manifest_path: str | None = Field(default=None, min_length=1)
    split_manifest_digest: str | None = Field(default=None, min_length=1)
    sampling_manifest_path: str | None = Field(default=None, min_length=1)
    sampling_manifest_digest: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def manifest_pairs(self):
        for name in ("presentation_manifest", "split_manifest", "sampling_manifest"):
            path = getattr(self, f"{name}_path")
            digest = getattr(self, f"{name}_digest")
            if (path is None) != (digest is None):
                raise ValueError(f"{name} path and digest must be supplied together")
        return self

    def manifest_reference(self, name: str) -> tuple[str, str] | None:
        path = getattr(self, f"{name}_path")
        digest = getattr(self, f"{name}_digest")
        return (path, digest) if path and digest else None

