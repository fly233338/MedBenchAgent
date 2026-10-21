"""Canonical source identity contracts used by preparation and planning."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(min_length=1)]
Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Source(Contract):
    id: Name
    version: Text | None = None
    kind: Literal["raw", "annotation"] = "raw"
    root: Text
    annotation: str = "cases.jsonl"


class SourcesArguments(Contract):
    sources: list[Source] = Field(min_length=1)
    dataset_id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
    feedback: str = ""

    @model_validator(mode="after")
    def unique_sources(self):
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("Source IDs must be unique within a dataset")
        return self


class DatasetAccessReceipt(Contract):
    """One access contract from Prepared indexing through compilation."""
    schema_version: Literal["dataset-access"] = "dataset-access"
    dataset_id: Text
    revision: Text
    cases_path: Text
    image_manifest_path: Text
    profile_path: Text
    duckdb_path: Text
    annotation_digest: Text
    image_manifest_digest: Text
    profile_digest: Text
    duckdb_digest: Text
    case_count: int = Field(ge=0)
    patient_count: int = Field(ge=0)
    mapping_path: Text | None = None
    mapping_digest: Text | None = None
    dataset_card_path: Text | None = None
    dataset_card_digest: Text | None = None
    bundle_root: Text | None = None
    split_manifest_path: Text | None = None
    split_manifest_digest: Text | None = None
    sampling_manifest_path: Text | None = None
    sampling_manifest_digest: Text | None = None

    @model_validator(mode="after")
    def paired_references(self):
        for name in ("mapping", "dataset_card", "split_manifest", "sampling_manifest"):
            if (getattr(self, f"{name}_path") is None) != (getattr(self, f"{name}_digest") is None):
                raise ValueError(f"{name} path and digest must be supplied together")
        return self

    def verify(self):
        import json
        from pathlib import Path
        from medbenchagent.data.sources import digest_file
        for name, digest_name in (("cases", "annotation"), ("image_manifest", "image_manifest"),
                                  ("profile", "profile"), ("duckdb", "duckdb"), ("mapping", "mapping"),
                                  ("dataset_card", "dataset_card"), ("split_manifest", "split_manifest"),
                                  ("sampling_manifest", "sampling_manifest")):
            path = getattr(self, f"{name}_path")
            if path is not None and (not Path(path).is_file() or digest_file(path) != getattr(self, f"{digest_name}_digest")):
                raise ValueError(f"Dataset access content changed or missing: {name}")
        profile = json.loads(Path(self.profile_path).read_text(encoding="utf-8"))
        if (profile["case_count"], profile["patient_count"]) != (self.case_count, self.patient_count):
            raise ValueError("Dataset access counts differ from profile")
        if self.dataset_card_path:
            card = json.loads(Path(self.dataset_card_path).read_text(encoding="utf-8"))
            if card["dataset_id"] != self.dataset_id:
                raise ValueError("Dataset Card belongs to another dataset")
        return self
