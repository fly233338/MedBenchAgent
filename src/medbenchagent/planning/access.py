"""Planning-time DatasetAccessReceipt contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetAccessReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["dataset-access"] = "dataset-access"
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    cases_path: str = Field(min_length=1)
    image_manifest_path: str = Field(min_length=1)
    profile_path: str = Field(min_length=1)
    duckdb_path: str = Field(min_length=1)
    annotation_digest: str = Field(min_length=1)
    image_manifest_digest: str = Field(min_length=1)
    profile_digest: str = Field(min_length=1)
    duckdb_digest: str = Field(min_length=1)
    case_count: int = Field(ge=0)
    patient_count: int = Field(ge=0)
    patient_identity: str = Field(min_length=1)
    mapping_path: str | None = None
    mapping_digest: str | None = None
    dataset_card_path: str | None = None
    dataset_card_digest: str | None = None
    source_schema_version: Literal["annotation"] = "annotation"
    source_access: dict[str, Any]
    source_root: str | None = None
    source_roots: dict[str, str] = Field(default_factory=dict)
    bundle_root: str | None = None
    split_manifest_path: str | None = Field(default=None, min_length=1)
    split_manifest_digest: str | None = Field(default=None, min_length=1)
    sampling_manifest_path: str | None = Field(default=None, min_length=1)
    sampling_manifest_digest: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def manifest_pairs(self):
        for name in ("split_manifest", "sampling_manifest", "mapping", "dataset_card"):
            path = getattr(self, f"{name}_path")
            digest = getattr(self, f"{name}_digest")
            if (path is None) != (digest is None):
                raise ValueError(f"{name} path and digest must be supplied together")
        return self

    @classmethod
    def bind(cls, dataset_id: str, access: dict[str, Any]):
        """Freeze the actual source access and profile facts into a checked binding."""
        from medbenchagent.data.sources import digest_file
        profile = json.loads(Path(access["profile_path"]).read_text(encoding="utf-8"))
        source = dict(access)
        for name in ("mapping", "dataset_card"):
            if source.get(f"{name}_path"):
                source[f"{name}_digest"] = digest_file(source[f"{name}_path"])
        values = {key: value for key, value in source.items()
                  if key in cls.model_fields and key not in {"schema_version", "dataset_id"}}
        receipt = cls.model_validate({**values, "dataset_id": dataset_id,
            "source_schema_version": source["schema_version"], "source_access": source,
            "case_count": profile["case_count"], "patient_count": profile["patient_count"],
            "patient_identity": "present" if profile["patient_count"] else "absent"})
        receipt.verified_source(dataset_id)
        return receipt

    def verified_source(self, dataset_id: str) -> dict[str, Any]:
        from medbenchagent.data.sources import digest_file
        if self.dataset_id != dataset_id:
            raise ValueError("Dataset access receipt belongs to another dataset")
        if self.source_access.get("dataset_id", dataset_id) != dataset_id:
            raise ValueError("Dataset access source belongs to another dataset")
        if self.source_access.get("schema_version") != self.source_schema_version:
            raise ValueError("Dataset access source schema mismatch")
        if self.source_access.get("revision") != self.revision:
            raise ValueError("Dataset access source revision mismatch")
        for name in ("source_root", "source_roots", "bundle_root"):
            if self.source_access.get(name, {} if name == "source_roots" else None) != getattr(self, name):
                raise ValueError(f"Dataset access binding mismatch: {name}")
        for name, digest_name in (("cases", "annotation"), ("image_manifest", "image_manifest"),
                                  ("profile", "profile"), ("duckdb", "duckdb"),
                                  ("mapping", "mapping"), ("dataset_card", "dataset_card"),
                                  ("split_manifest", "split_manifest"), ("sampling_manifest", "sampling_manifest")):
            path = getattr(self, f"{name}_path")
            expected = getattr(self, f"{digest_name}_digest")
            if self.source_access.get(f"{name}_path") != path or self.source_access.get(f"{digest_name}_digest") != expected:
                raise ValueError(f"Dataset access binding mismatch: {name}")
            if path is not None and (not Path(path).is_file() or digest_file(path) != expected):
                raise ValueError(f"Dataset access content changed or missing: {name}")
        profile = json.loads(Path(self.profile_path).read_text(encoding="utf-8"))
        if (profile["case_count"], profile["patient_count"]) != (self.case_count, self.patient_count):
            raise ValueError("Dataset access counts differ from profile")
        if self.patient_identity != ("present" if self.patient_count else "absent"):
            raise ValueError("Dataset access patient identity differs from profile")
        if self.dataset_card_path:
            card = json.loads(Path(self.dataset_card_path).read_text(encoding="utf-8"))
            if card["dataset_id"] != dataset_id:
                raise ValueError("Dataset Card belongs to another dataset")
        return dict(self.source_access)

    def manifest_reference(self, name: str) -> tuple[str, str] | None:
        path = getattr(self, f"{name}_path")
        digest = getattr(self, f"{name}_digest")
        return (path, digest) if path and digest else None

