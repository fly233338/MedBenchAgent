"""Profile evidence and prepared, backend-scoped Case collections."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .dataset import DatasetMember, DatasetRef, DatasetSource, CasePage, _relative_file, file_digest
from .models import CanonicalModel, CaseRecord, SourceRef


class ProductBinding(CanonicalModel):
    artifact_id: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceManifest(CanonicalModel):
    dataset: DatasetRef
    sources: list[DatasetSource] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self):
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("Source IDs must be unique")
        return self


class ProfileObservation(CanonicalModel):
    observation_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_refs: list[SourceRef] = Field(min_length=1)


class DatasetProfile(SourceManifest):
    description: str = Field(min_length=1)
    fields: list[dict[str, Any]] = Field(default_factory=list)
    assets: dict[str, Any] = Field(default_factory=dict)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    coverage: Literal["sampled", "complete", "unknown"] = "unknown"
    inspected_scope: str = Field(min_length=1)
    observations: list[ProfileObservation] = Field(min_length=1)
    unknowns: list[str] = Field(default_factory=list)
    samples: list[CaseRecord] = Field(default_factory=list)
    members: list[DatasetMember] = Field(default_factory=list)

    @model_validator(mode="after")
    def identities(self):
        for values, key in ((self.observations, "observation_id"), (self.samples, "case_id")):
            if len({getattr(value, key) for value in values}) != len(values):
                raise ValueError(f"Duplicate Profile {key}")
        return self


class PreparedAsset(CanonicalModel):
    asset_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    file_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str = Field(min_length=1)
    format: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskField(CanonicalModel):
    name: str = Field(min_length=1)
    type: Literal["asset", "text", "value"]
    value: Any
    input_visibility: Literal["model_visible", "reference_only", "unknown"] = "unknown"
    source_refs: list[SourceRef] = Field(min_length=1)
    ref: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self):
        return self.name

    @model_validator(mode="after")
    def values(self):
        json.dumps(self.value, allow_nan=False)
        if self.type == "text" and not isinstance(self.value, str):
            raise ValueError("Text fields require a string")
        if self.type == "asset":
            values = self.value if isinstance(self.value, list) else [self.value]
            if not values or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("Asset fields require an Asset ID or an ordered list of Asset IDs")
        return self


class TaskCase(CanonicalModel):
    case_id: str = Field(min_length=1)
    scenario_ids: list[str] = Field(min_length=1)
    fields: list[TaskField] = Field(min_length=1)
    source_refs: list[SourceRef] = Field(min_length=1)

    @property
    def evidence(self):
        return self.fields

    @model_validator(mode="after")
    def identities(self):
        names = {field.name for field in self.fields}
        if len(names) != len(self.fields) or len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError("Task Case fields and Scenario IDs must be unique")
        if any(set(field.ref) - names for field in self.fields):
            raise ValueError("Task field references an unknown field")
        return self


class TaskDataManifest(SourceManifest):
    backend_id: str = Field(min_length=1)
    scenario_ids: list[str] = Field(min_length=1)
    requirement_ids: list[str] = Field(default_factory=list)
    profile: ProductBinding | None = None
    cases_path: str = "cases.jsonl"
    assets_path: str = "assets.jsonl"
    members: list[DatasetMember] = Field(default_factory=list)
    case_count: int = Field(default=0, ge=0)
    scenario_counts: dict[str, int] = Field(default_factory=dict)


def _rows(path, schema):
    with path.open(encoding="utf-8") as stream:
        return [schema.model_validate_json(line) for line in stream if line.strip()]


def _source_roots(root, sources):
    return {s.source_id: (root / s.root).resolve() for s in sources}


def _members(root, sources, references):
    roots = _source_roots(root, sources)
    return [
        DatasetMember(
            source_id=source_id,
            path=path,
            digest=file_digest(
                _relative_file(root if source_id is None else roots[source_id], path)
            ),
        )
        for source_id, path in sorted(references, key=lambda pair: (pair[0] or "", pair[1]))
    ]


def _profile_references(profile):
    refs = [ref for observation in profile.observations for ref in observation.source_refs]
    refs += [
        ref
        for case in profile.samples
        for evidence in case.evidence
        for ref in evidence.source_refs
    ]
    refs += [
        ref
        for case in profile.samples
        for relation in case.relations
        for ref in relation.source_refs
    ]
    return {(ref.source_id, ref.path) for ref in refs}


def seal_profile(profile: DatasetProfile, root: Path):
    return profile.model_copy(
        update={"members": _members(root, profile.sources, _profile_references(profile))}
    )


def _task_references(manifest, cases, assets):
    refs = {(None, manifest.cases_path), (None, manifest.assets_path)}
    refs.update((asset.source_id, asset.source_path) for asset in assets)
    for case in cases:
        refs.update((ref.source_id, ref.path) for ref in case.source_refs)
        refs.update((ref.source_id, ref.path) for field in case.fields for ref in field.source_refs)
    return refs


def seal_task_data(manifest: TaskDataManifest, root: Path):
    cases = _rows(_relative_file(root, manifest.cases_path), TaskCase)
    assets = _rows(_relative_file(root, manifest.assets_path), PreparedAsset)
    counts = {key: sum(key in case.scenario_ids for case in cases) for key in manifest.scenario_ids}
    return manifest.model_copy(
        update={
            "case_count": len(cases),
            "scenario_counts": counts,
            "members": _members(root, manifest.sources, _task_references(manifest, cases, assets)),
        }
    )


def _verify_members(root, product, required):
    keys = {(m.source_id, m.path) for m in product.members}
    if keys != required or len(keys) != len(product.members):
        raise ValueError("Product members must exactly cover its referenced files")
    roots = _source_roots(root, product.sources)
    for member in product.members:
        path = _relative_file(
            root if member.source_id is None else roots[member.source_id], member.path
        )
        if file_digest(path) != member.digest:
            raise ValueError(f"Product member changed: {member.path}")


class ProfileReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.profile = DatasetProfile.model_validate_json(self.path.read_bytes())
        self._digest = file_digest(self.path)
        self.verify_integrity()

    def verify_integrity(self):
        if file_digest(self.path) != self._digest:
            raise ValueError("Profile changed")
        _verify_members(self.path.parent, self.profile, _profile_references(self.profile))

    @property
    def content_digest(self):
        self.verify_integrity()
        return self._digest

    def get_manifest(self):
        self.verify_integrity()
        return self.profile.dataset.model_copy(deep=True)

    def get_observation(self, identifier):
        self.verify_integrity()
        for observation in self.profile.observations:
            if observation.observation_id == identifier:
                return observation
        raise KeyError(f"Unknown Profile observation: {identifier}")

    def list_cases(self, offset=0, limit=100):
        self.verify_integrity()
        if offset < 0 or limit < 1:
            raise ValueError("Invalid page")
        end = offset + limit
        return CasePage(
            items=self.profile.samples[offset:end],
            next_offset=end if end < len(self.profile.samples) else None,
        )

    def get_case(self, identifier):
        self.verify_integrity()
        for case in self.profile.samples:
            if case.case_id == identifier:
                return case.model_copy(deep=True)
        raise KeyError(f"Profile has no inspected sample Case: {identifier}")

    def get_evidence(self, case_id, evidence_id):
        for evidence in self.get_case(case_id).evidence:
            if evidence.id == evidence_id:
                return evidence
        raise KeyError(evidence_id)

    def read_mapping(self):
        self.verify_integrity()
        return self.profile.model_dump(mode="json", exclude={"members", "samples"})


class TaskDataReader:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = TaskDataManifest.model_validate_json(self.path.read_bytes())
        self._digest = file_digest(self.path)
        self.cases = _rows(_relative_file(self.path.parent, self.manifest.cases_path), TaskCase)
        assets = _rows(_relative_file(self.path.parent, self.manifest.assets_path), PreparedAsset)
        self.assets = {asset.asset_id: asset for asset in assets}
        if len(self.assets) != len(assets) or len({c.case_id for c in self.cases}) != len(
            self.cases
        ):
            raise ValueError("TaskData Case and Asset IDs must be unique")
        if len(self.manifest.scenario_ids) != len(set(self.manifest.scenario_ids)):
            raise ValueError("TaskData Scenario IDs must be unique")
        for case in self.cases:
            if set(case.scenario_ids) - set(self.manifest.scenario_ids):
                raise ValueError("Task Case references a Scenario outside the Manifest")
            for field in case.fields:
                if field.type == "asset":
                    for key in field.value if isinstance(field.value, list) else [field.value]:
                        if key not in self.assets:
                            raise ValueError(f"Unknown prepared Asset: {key}")
        counts = {
            key: sum(key in c.scenario_ids for c in self.cases)
            for key in self.manifest.scenario_ids
        }
        if self.manifest.case_count != len(self.cases) or self.manifest.scenario_counts != counts:
            raise ValueError("TaskData coverage counts do not match its actual Cases")
        self.verify_integrity()

    def verify_integrity(self):
        if file_digest(self.path) != self._digest:
            raise ValueError("TaskData Manifest changed")
        _verify_members(
            self.path.parent,
            self.manifest,
            _task_references(self.manifest, self.cases, list(self.assets.values())),
        )
        accepted = {(m.source_id, m.path): m.digest for m in self.manifest.members}
        for asset in self.assets.values():
            if accepted[(asset.source_id, asset.source_path)] != asset.file_digest:
                raise ValueError(
                    f"Asset entry digest differs from the sealed member: {asset.asset_id}"
                )

    @property
    def content_digest(self):
        self.verify_integrity()
        return self._digest

    def get_case(self, identifier):
        # Operation boundaries verify the complete snapshot. Per-Case reads recheck records;
        # selected resource bytes are verified by asset_value without rescanning every image.
        if file_digest(self.path) != self._digest:
            raise ValueError("TaskData Manifest changed")
        for member in self.manifest.members:
            if (
                member.source_id is None
                and file_digest(_relative_file(self.path.parent, member.path)) != member.digest
            ):
                raise ValueError(f"TaskData records changed: {member.path}")
        for case in self.cases:
            if case.case_id == identifier:
                return case.model_copy(deep=True)
        raise KeyError(f"Unknown Task Case: {identifier}")

    def get_evidence(self, case_id, field_name):
        for field in self.get_case(case_id).fields:
            if field.name == field_name:
                return field
        raise KeyError(field_name)

    def asset_value(self, asset_id):
        asset = self.assets[asset_id]
        roots = _source_roots(self.path.parent, self.manifest.sources)
        path = _relative_file(roots[asset.source_id], asset.source_path)
        if file_digest(path) != asset.file_digest:
            raise ValueError(f"Prepared Asset changed: {asset_id}")
        return {
            "asset_id": asset_id,
            "path": str(path),
            "file_digest": asset.file_digest,
            "kind": asset.kind,
            "format": asset.format,
        }


def write_once(path: Path, content: bytes):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Cannot overwrite immutable product: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def snapshot_task_data(source: Path, destination: Path):
    reader = TaskDataReader(source)
    manifest = reader.manifest
    roots = _source_roots(source.parent, manifest.sources)
    destination = destination.resolve()
    for member in manifest.members:
        root = source.parent if member.source_id is None else roots[member.source_id]
        relative = (
            member.path if member.source_id is None else f"sources/{member.source_id}/{member.path}"
        )
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("TaskData member escapes destination")
        content = _relative_file(root, member.path).read_bytes()
        if hashlib.sha256(content).hexdigest() != member.digest:
            raise ValueError("TaskData member changed during snapshot")
        write_once(target, content)
    copied = manifest.model_copy(
        update={
            "sources": [
                DatasetSource(source_id=s.source_id, root=f"sources/{s.source_id}")
                for s in manifest.sources
            ]
        }
    )
    target = destination / "manifest.json"
    write_once(target, (copied.model_dump_json(indent=2) + "\n").encode())
    TaskDataReader(target)
    return target
