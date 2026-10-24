"""A small Case-based file reader, independent of Prepared, DuckDB and planning."""

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import Field

from .models import AssetEntry, CanonicalModel, CaseRecord, Evidence, SourceRef, validate_case


class DatasetRef(CanonicalModel):
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class DatasetSource(CanonicalModel):
    source_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
    root: str = Field(min_length=1)


class DatasetMember(CanonicalModel):
    source_id: str | None = None
    path: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetManifest(DatasetRef):
    cases_path: str = Field(min_length=1)
    assets_path: str = Field(min_length=1)
    sources: list[DatasetSource] = Field(default_factory=list)
    mapping_path: str | None = None
    sampling_path: str | None = None
    split_path: str | None = None
    members: list[DatasetMember] = Field(default_factory=list)


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _referenced_members(root, manifest):
    paths = {
        (None, value)
        for value in (
            manifest.cases_path,
            manifest.assets_path,
            manifest.mapping_path,
            manifest.sampling_path,
            manifest.split_path,
        )
        if value
    }
    assets = _rows(_relative_file(root, manifest.assets_path), AssetEntry)
    cases = _rows(_relative_file(root, manifest.cases_path), CaseRecord)
    paths.update((asset.source_id, asset.source_path) for asset in assets)
    for case in cases:
        for owner in [*case.evidence, *case.relations]:
            paths.update((ref.source_id, ref.path) for ref in owner.source_refs)
    return paths


def seal_manifest(root: Path, manifest: DatasetManifest) -> DatasetManifest:
    """Explicit producer operation. Readers never recompute accepted identities."""
    root = root.resolve()
    sources = {source.source_id: (root / source.root).resolve() for source in manifest.sources}
    members = [
        DatasetMember(
            source_id=source,
            path=path,
            digest=file_digest(_relative_file(root if source is None else sources[source], path)),
        )
        for source, path in sorted(
            _referenced_members(root, manifest),
            key=lambda item: (item[0] or "", item[1]),
        )
    ]
    return manifest.model_copy(update={"members": members})


def snapshot_dataset(source: Path, destination: Path) -> Path:
    """Copy a verified canonical dataset into a self-contained, write-once snapshot."""
    reader = FileDatasetReader(source)
    manifest = reader.get_manifest()
    destination = destination.resolve()

    def copy_once(path, content):
        target = (destination / path).resolve()
        if not target.is_relative_to(destination):
            raise ValueError("Dataset snapshot path escapes its destination")
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"Cannot overwrite dataset member: {path}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)

    for member in manifest.members:
        root = reader._root if member.source_id is None else reader._source_roots[member.source_id]
        relative = (
            member.path if member.source_id is None else f"sources/{member.source_id}/{member.path}"
        )
        content = _relative_file(root, member.path).read_bytes()
        if hashlib.sha256(content).hexdigest() != member.digest:
            raise ValueError("Dataset member changed during snapshot")
        copy_once(relative, content)
    frozen = manifest.model_copy(
        update={
            "sources": [
                DatasetSource(source_id=item.source_id, root=f"sources/{item.source_id}")
                for item in manifest.sources
            ]
        }
    )
    copy_once("manifest.json", (frozen.model_dump_json(indent=2) + "\n").encode("utf-8"))
    FileDatasetReader(destination / "manifest.json")
    return destination / "manifest.json"


class CasePage(CanonicalModel):
    items: list[CaseRecord]
    next_offset: int | None = None


class ResolvedLocator(CanonicalModel):
    source_id: str
    path: str
    file_digest: str


class DatasetReader(Protocol):
    def get_manifest(self) -> DatasetManifest: ...
    def list_cases(self, offset: int = 0, limit: int = 100) -> CasePage: ...
    def get_case(self, case_id: str) -> CaseRecord: ...
    def get_evidence(self, case_id: str, evidence_id: str) -> Evidence: ...
    def get_asset(self, asset_id: str) -> AssetEntry: ...
    def resolve_asset(self, asset_id: str) -> ResolvedLocator: ...


def _relative_file(root: Path, value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or path.drive:
        raise ValueError(f"Expected a relative file path: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"File escapes its declared root: {value}")

    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


Record = TypeVar("Record", bound=CanonicalModel)


def _rows(path: Path, model: type[Record]) -> list[Record]:
    with path.open(encoding="utf-8") as stream:
        return [model.model_validate_json(line) for line in stream if line.strip()]


def _unique(rows, key, label):
    indexed = {}
    for row in rows:
        identifier = getattr(row, key)
        if identifier in indexed:
            raise ValueError(f"Duplicate {label}: {identifier}")
        indexed[identifier] = row
    return indexed


class FileDatasetReader:
    """Read a small JSON manifest and JSONL records as one in-memory snapshot."""

    def __init__(self, manifest_path: str | Path):
        path = Path(manifest_path).resolve()
        self._path = path
        self._digest = file_digest(path)
        self._initializing = True
        self._root = path.parent
        self._manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        self._sources = _unique(self._manifest.sources, "source_id", "Source ID")
        self._source_roots = {}
        for source in self._sources.values():
            root = (self._root / source.root).resolve()
            if not root.is_dir():
                raise FileNotFoundError(root)
            self._source_roots[source.source_id] = root
        for value in (
            self._manifest.mapping_path,
            self._manifest.sampling_path,
            self._manifest.split_path,
        ):
            if value is not None:
                _relative_file(self._root, value)
        self._assets = _unique(
            _rows(_relative_file(self._root, self._manifest.assets_path), AssetEntry),
            "asset_id",
            "Asset ID",
        )
        self._cases = _unique(
            _rows(_relative_file(self._root, self._manifest.cases_path), CaseRecord),
            "case_id",
            "Case ID",
        )
        self._case_ids = sorted(self._cases)
        accepted = {(member.source_id, member.path) for member in self._manifest.members}
        if len(accepted) != len(self._manifest.members) or accepted != _referenced_members(
            self._root,
            self._manifest,
        ):
            raise ValueError(
                "Dataset members must cover exactly the referenced files without duplicates"
            )
        for asset in self._assets.values():
            self.resolve_asset(asset.asset_id)
        for case in self._cases.values():
            validate_case(case, self._assets)
            for evidence in case.evidence:
                # Mask assets must also resolve; no special locator fallback.
                if evidence.type == "asset" and evidence.value not in self._assets:
                    raise ValueError(f"Unknown Asset ID: {evidence.value}")
                for reference in evidence.source_refs:
                    self._check_source(reference)
            for relation in case.relations:
                for reference in relation.source_refs:
                    self._check_source(reference)
        self._initializing = False
        self.verify_integrity()

    def verify_integrity(self):
        if self._initializing:
            return

        if file_digest(self._path) != self._digest:
            raise ValueError("Dataset manifest content changed")
        for member in self._manifest.members:
            root = self._root if member.source_id is None else self._source_roots[member.source_id]
            if file_digest(_relative_file(root, member.path)) != member.digest:
                raise ValueError(f"Dataset member content changed: {member.path}")

    @property
    def content_digest(self):
        self.verify_integrity()
        return self._digest

    def read_mapping(self):
        self.verify_integrity()
        if self._manifest.mapping_path is None:
            return None
        return json.loads(
            _relative_file(self._root, self._manifest.mapping_path).read_text(encoding="utf-8")
        )

    def _check_source(self, reference: SourceRef) -> None:
        if reference.source_id not in self._source_roots:
            raise ValueError(f"Unknown Source ID: {reference.source_id}")
        _relative_file(self._source_roots[reference.source_id], reference.path)

    def get_manifest(self) -> DatasetManifest:
        self.verify_integrity()
        return self._manifest.model_copy(deep=True)

    def list_cases(self, offset: int = 0, limit: int = 100) -> CasePage:
        self.verify_integrity()
        if type(offset) is not int or type(limit) is not int or offset < 0 or limit <= 0:
            raise ValueError("offset must be nonnegative and limit must be positive integers")
        ids = self._case_ids[offset : offset + limit]
        end = offset + len(ids)
        return CasePage(
            items=[self._cases[identifier].model_copy(deep=True) for identifier in ids],
            next_offset=end if end < len(self._case_ids) else None,
        )

    def get_case(self, case_id: str) -> CaseRecord:
        self.verify_integrity()
        return self._cases[case_id].model_copy(deep=True)

    def get_evidence(self, case_id: str, evidence_id: str) -> Evidence:
        self.verify_integrity()
        for evidence in self._cases[case_id].evidence:
            if evidence.id == evidence_id:
                return evidence.model_copy(deep=True)
        raise KeyError(f"Unknown Evidence: {case_id}/{evidence_id}")

    def get_asset(self, asset_id: str) -> AssetEntry:
        self.verify_integrity()
        return self._assets[asset_id].model_copy(deep=True)

    def resolve_asset(self, asset_id: str) -> ResolvedLocator:
        asset = self._assets[asset_id]
        if asset.source_id not in self._source_roots:
            raise ValueError(f"Unknown Source ID: {asset.source_id}")
        path = _relative_file(self._source_roots[asset.source_id], asset.source_path)
        if file_digest(path) != asset.file_digest:
            raise ValueError(f"Asset content changed: {asset_id}")
        return ResolvedLocator(
            source_id=asset.source_id,
            path=str(path),
            file_digest=asset.file_digest,
        )
