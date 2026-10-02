"""Planning-time DatasetAccessReceipt contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DatasetAccessReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = "dataset-access.v2"
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
