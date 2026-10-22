"""A small Case-based file reader, independent of Prepared, DuckDB and planning."""

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import Field

from .models import AssetEntry, CanonicalModel, CaseRecord, Evidence, SourceRef, validate_case


class DatasetRef(CanonicalModel):
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class DatasetSource(CanonicalModel):
    source_id: str = Field(min_length=1)
    root: str = Field(min_length=1)


class DatasetManifest(DatasetRef):
    cases_path: str = Field(min_length=1)
    assets_path: str = Field(min_length=1)
    sources: list[DatasetSource] = Field(default_factory=list)
    mapping_path: str | None = None
    sampling_path: str | None = None
    split_path: str | None = None


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
            self._manifest.mapping_path, self._manifest.sampling_path, self._manifest.split_path,
        ):
            if value is not None:
                _relative_file(self._root, value)
        self._assets = _unique(
            _rows(_relative_file(self._root, self._manifest.assets_path), AssetEntry),
            "asset_id", "Asset ID",
        )
        self._cases = _unique(
            _rows(_relative_file(self._root, self._manifest.cases_path), CaseRecord),
            "case_id", "Case ID",
        )
        self._case_ids = sorted(self._cases)
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

    def _check_source(self, reference: SourceRef) -> None:
        if reference.source_id not in self._source_roots:
            raise ValueError(f"Unknown Source ID: {reference.source_id}")
        _relative_file(self._source_roots[reference.source_id], reference.path)

    def get_manifest(self) -> DatasetManifest:
        return self._manifest.model_copy(deep=True)

    def list_cases(self, offset: int = 0, limit: int = 100) -> CasePage:
        if type(offset) is not int or type(limit) is not int or offset < 0 or limit <= 0:
            raise ValueError("offset must be nonnegative and limit must be positive integers")
        ids = self._case_ids[offset:offset + limit]
        end = offset + len(ids)
        return CasePage(
            items=[self.get_case(identifier) for identifier in ids],
            next_offset=end if end < len(self._case_ids) else None,
        )

    def get_case(self, case_id: str) -> CaseRecord:
        return self._cases[case_id].model_copy(deep=True)

    def get_evidence(self, case_id: str, evidence_id: str) -> Evidence:
        for evidence in self._cases[case_id].evidence:
            if evidence.id == evidence_id:
                return evidence.model_copy(deep=True)
        raise KeyError(f"Unknown Evidence: {case_id}/{evidence_id}")

    def get_asset(self, asset_id: str) -> AssetEntry:
        return self._assets[asset_id].model_copy(deep=True)

    def resolve_asset(self, asset_id: str) -> ResolvedLocator:
        asset = self._assets[asset_id]
        if asset.source_id not in self._source_roots:
            raise ValueError(f"Unknown Source ID: {asset.source_id}")
        path = _relative_file(self._source_roots[asset.source_id], asset.source_path)
        return ResolvedLocator(
            source_id=asset.source_id, path=str(path), file_digest=asset.file_digest,
        )
