"""Experiment-level Benchmark Intermediate Representation (BIR).

The BIR is the canonical, resumable state for one experiment.  Run folders may
contain projections of this state, but agents must use :class:`BIRStore` for
reads and writes so revisions, digests and dependencies are checked in one
place.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from filelock import FileLock
from medbenchagent.planning.extensions import PlanningExtensionCatalog, load_extension_catalog


class BIRConflictError(ValueError):
    """Raised when a writer is based on an older BIR revision."""


@dataclass(frozen=True)
class BIRCommit:
    revision: int
    manifest_digest: str
    changed_paths: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class BIRValidationReport:
    valid: bool
    revision: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BIRTemplate:
    path: Path
    template_revision: str
    digest: str
    files: tuple[str, ...]


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_file(path: str | Path) -> str:
    return _digest_bytes(Path(path).read_bytes())


def digest_tree(root: str | Path, *, exclude: Iterable[str] = ()) -> str:
    """Digest a directory deterministically by sorted POSIX path and bytes."""
    root = Path(root)
    excluded = {PurePosixPath(item).as_posix() for item in exclude}
    hasher = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()),
                       key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or relative.startswith(".staging/"):
            continue
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def _safe_relative(relative_path: str) -> PurePosixPath:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("BIR path must be a non-empty relative POSIX path")
    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or any(part in {"", ".", ".."} for part in normalized.parts):
        raise ValueError(f"Unsafe BIR path: {relative_path}")
    return normalized


def _schema(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("schema_version", "json"))
    return "text"


class RegistrySnapshot:
    """Loaded, immutable view of the experiment Registry."""

    def __init__(self, root: Path, documents: dict[str, Any], digests: dict[str, str],
                 prompts: dict[str, str], extensions: PlanningExtensionCatalog):
        self.root = root
        self.documents = documents
        self.digests = digests
        self.prompts = prompts
        self.extensions = extensions

    @classmethod
    def from_directory(cls, root: str | Path) -> "RegistrySnapshot":
        root = Path(root)
        required = (
            "axes.json", "capability_categories.json", "scenario_constraints.json",
            "task_contracts.json", "resolver_policy.json",
        )
        documents: dict[str, Any] = {}
        digests: dict[str, str] = {}
        for name in required:
            path = root / name
            if not path.is_file():
                raise ValueError(f"BIR Registry file is missing: {path}")
            try:
                documents[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot read BIR Registry file {path}: {exc}") from exc
            digests[name] = digest_file(path)
        prompt_root = root / "agent_prompts"
        prompts = {}
        if prompt_root.is_dir():
            for path in sorted(prompt_root.glob("*.md")):
                prompts[path.stem] = path.read_text(encoding="utf-8")
                digests[f"agent_prompts/{path.name}"] = digest_file(path)
        extensions = load_extension_catalog(root / "planning_extensions")
        for extension in extensions:
            digests[f"planning_extensions/{extension.path}"] = extension.digest
        snapshot = cls(root, documents, digests, prompts, extensions)
        for operation in snapshot.axis_values("operation"):
            snapshot.operation_definition(operation)
        return snapshot

    @property
    def digest(self) -> str:
        payload = "".join(f"{key}:{self.digests[key]}\n" for key in sorted(self.digests)).encode("utf-8")
        return _digest_bytes(payload)

    @property
    def axis_order(self) -> tuple[str, ...]:
        document = self.documents["axes.json"]
        return tuple(document.get("axis_order") or [row["id"] for row in document.get("axes", [])])

    def axis_values(self, axis: str) -> tuple[str, ...]:
        for row in self.documents["axes.json"].get("axes", []):
            if row.get("id") == axis:
                return tuple(value["id"] if isinstance(value, dict) else value for value in row.get("values", []))
        raise ValueError(f"Unknown BIR Axis: {axis}")

    def axis_definitions(self) -> list[dict[str, Any]]:
        return list(self.documents["axes.json"].get("axes", []))

    def operation_definition(self, operation: str) -> dict[str, Any]:
        operation_axis = next((axis for axis in self.axis_definitions() if axis.get("id") == "operation"), None)
        for value in (operation_axis or {}).get("values", []):
            if value.get("id") == operation:
                mode = value.get("answer_options")
                if mode not in {"required", "forbidden"}:
                    raise ValueError(f"Operation {operation!r} has invalid answer_options: {mode!r}")
                return dict(value)
        raise ValueError(f"Unknown BIR operation: {operation!r}")

    def answer_options_mode(self, operation: str) -> str:
        return self.operation_definition(operation)["answer_options"]

    @property
    def capability_by_operation(self) -> dict[str, str]:
        return dict(self.documents["capability_categories.json"].get("operation_to_category", {}))

    @property
    def capability_categories(self) -> tuple[str, ...]:
        return tuple(row["id"] for row in self.documents["capability_categories.json"].get("categories", []))

    @property
    def scenario_constraints(self) -> dict[str, Any]:
        return self.documents["scenario_constraints.json"]

    @property
    def task_contracts(self) -> dict[str, Any]:
        return self.documents["task_contracts.json"]

    @property
    def resolver_policy(self) -> dict[str, Any]:
        return self.documents["resolver_policy.json"]

    def prompt(self, name: str) -> str:
        try:
            return self.prompts[name]
        except KeyError as exc:
            raise ValueError(f"BIR prompt is missing: {name}") from exc

    def planning_extensions(self) -> PlanningExtensionCatalog:
        return self.extensions

    def validate_value(self, axis: str, value: str) -> None:
        if value not in self.axis_values(axis):
            raise ValueError(f"Unknown value {value!r} for BIR Axis {axis}")


class BIRStore:
    """Safe, revisioned access to an experiment's BIR directory."""

    SCHEMA_VERSION = "bir"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.lock = FileLock(str(self.root / ".bir.lock"))
        if not self.root.is_dir():
            raise ValueError(f"BIR directory does not exist: {self.root}")
        self.recover()

    @classmethod
    def load_template(cls, template_dir: str | Path) -> BIRTemplate:
        path = Path(template_dir).resolve()
        if not path.is_dir():
            raise ValueError(f"BIR template directory does not exist: {path}")
        for required in ("manifest.json", "benchmark_info.json", "allocation.json", "delivery.json"):
            if not (path / required).is_file():
                raise ValueError(f"BIR template requires {required}")
        RegistrySnapshot.from_directory(path / "registry")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("BIR template requires manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        template_revision = str(manifest.get("template_revision") or manifest.get("revision") or "1")
        for file in path.rglob("*"):
            if not file.is_file() or file.name == ".keep":
                continue
            if file.suffix in {".json", ".jsonl", ".toml"}:
                try:
                    value = json.loads(file.read_text(encoding="utf-8")) if file.suffix == ".json" else None
                except ValueError as exc:
                    raise ValueError(f"Invalid BIR template JSON: {file}: {exc}") from exc
                if value is not None and _contains_absolute_path(value):
                    raise ValueError(f"BIR template cannot contain an absolute path: {file}")
        files = tuple(sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()))
        return BIRTemplate(path, template_revision, digest_tree(path), files)

    @classmethod
    def initialize(cls, target_dir: str | Path, template_dir: str | Path, experiment_id: str) -> "BIRStore":
        target = Path(target_dir).resolve()
        if target.exists():
            raise ValueError(f"BIR directory already exists: {target}")
        template = cls.load_template(template_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(template.path, target)
        (target / "datasets").mkdir(exist_ok=True)
        (target / "outputs").mkdir(exist_ok=True)
        store = cls.__new__(cls)
        store.root = target
        store.lock = FileLock(str(target / ".bir.lock"))
        manifest = {
            "schema_version": cls.SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "template_revision": template.template_revision,
            "template_digest": template.digest,
            "revision": 0,
            "status": "empty",
            "parts": {},
        }
        store._write_manifest(manifest)
        store._refresh_parts(manifest)
        store._write_manifest(manifest)
        return store

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _read_manifest(self) -> dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid BIR manifest: {self.manifest_path}: {exc}") from exc
        if value.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported BIR schema: {value.get('schema_version')}")
        return value

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        _atomic_replace(self.manifest_path, _canonical_bytes(manifest))

    def _path(self, relative_path: str) -> Path:
        relative = _safe_relative(relative_path)
        path = (self.root / Path(*relative.parts)).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"BIR path escapes root: {relative_path}")
        return path

    def _refresh_parts(self, manifest: dict[str, Any], *, revision: int | None = None,
                       changed_paths: Iterable[str] = ()) -> None:
        changed = {PurePosixPath(item).as_posix() for item in changed_paths}
        parts: dict[str, Any] = {}
        for path in sorted(item for item in self.root.rglob("*") if item.is_file() and item.name != ".bir.lock"):
            relative = path.relative_to(self.root).as_posix()
            if relative in {"manifest.json"} or relative.startswith(".staging/"):
                continue
            schema = "text"
            if path.suffix == ".json":
                try:
                    schema = _schema(json.loads(path.read_text(encoding="utf-8")))
                except ValueError:
                    schema = "invalid"
            previous = manifest.get("parts", {}).get(relative, {})
            parts[relative] = {
                "schema_version": schema,
                "digest": digest_file(path),
                "revision": revision if relative in changed and revision is not None else previous.get("revision", 0),
                "dependencies": previous.get("dependencies", {}),
            }
        manifest["parts"] = parts

    def load(self, relative_path: str, *, expected_schema: str | None = None,
             verify_digest: bool = True) -> dict[str, Any]:
        path = self._path(relative_path)
        if not path.is_file():
            raise ValueError(f"BIR part does not exist: {relative_path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if expected_schema and value.get("schema_version") != expected_schema:
            raise ValueError(f"BIR part {relative_path} has unexpected schema")
        if verify_digest:
            part = self._read_manifest().get("parts", {}).get(PurePosixPath(relative_path).as_posix())
            if part and part.get("digest") != digest_file(path):
                raise ValueError(f"BIR part digest mismatch: {relative_path}")
        return value

    def registry(self) -> RegistrySnapshot:
        return RegistrySnapshot.from_directory(self.root / "registry")

    def digest(self, relative_path: str) -> str:
        return digest_file(self._path(relative_path))

    def validate(self, *, scope: str = "full", expected_revision: int | None = None) -> BIRValidationReport:
        manifest = self._read_manifest()
        errors: list[str] = []
        warnings: list[str] = []
        if not manifest.get("template_revision") or not manifest.get("template_digest"):
            errors.append("BIR manifest is missing frozen template metadata")
        for required in ("benchmark_info.json", "allocation.json", "delivery.json"):
            if not (self.root / required).is_file():
                errors.append(f"Missing required BIR part: {required}")
        if expected_revision is not None and manifest.get("revision") != expected_revision:
            errors.append(f"Expected BIR revision {expected_revision}, found {manifest.get('revision')}")
        for relative, part in manifest.get("parts", {}).items():
            path = self._path(relative)
            if not path.is_file():
                errors.append(f"Missing BIR part: {relative}")
                continue
            if part.get("digest") != digest_file(path):
                errors.append(f"Digest mismatch: {relative}")
            if path.suffix == ".json":
                parsed = None
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except ValueError as exc:
                    errors.append(f"Invalid JSON in {relative}: {exc}")
                if relative.startswith("registry/") and parsed is not None and _contains_absolute_path(parsed):
                    errors.append(f"Registry contains an absolute path: {relative}")
        if scope == "full":
            try:
                self.registry()
            except ValueError as exc:
                errors.append(str(exc))
        return BIRValidationReport(not errors, int(manifest.get("revision", 0)), tuple(errors), tuple(warnings))

    def update(self, relative_path: str, value: dict[str, Any], *, run_id: str,
               actor: str, expected_revision: int | None = None, dependencies: dict[str, str] | None = None,
               expected_digest: str | None = None) -> BIRCommit:
        return self.transaction({relative_path: value}, run_id=run_id, actor=actor,
                                 expected_revision=expected_revision, dependencies=dependencies,
                                 expected_digests={relative_path: expected_digest} if expected_digest else None)

    def transaction(self, changes: dict[str, dict[str, Any]], *, run_id: str, actor: str,
                    expected_revision: int | None = None, dependencies: dict[str, str] | None = None,
                    expected_digests: dict[str, str] | None = None) -> BIRCommit:
        if not changes:
            manifest = self._read_manifest()
            return BIRCommit(int(manifest["revision"]), digest_file(self.manifest_path), (), manifest.get("status", "empty"))
        with self.lock:
            manifest = self._read_manifest()
            actual = int(manifest.get("revision", 0))
            if expected_revision is not None and actual != expected_revision:
                raise ValueError(f"BIR revision conflict: expected {expected_revision}, found {actual}")
            for relative, expected_digest in (expected_digests or {}).items():
                if not expected_digest:
                    continue
                path = self._path(relative)
                if not path.is_file() or digest_file(path) != expected_digest:
                    raise ValueError(f"BIR digest conflict: {relative}")
            registry_changes = [path for path in changes if path.replace("\\", "/").startswith("registry/")]
            if registry_changes:
                raise ValueError("BIR Registry is immutable after experiment creation")
            staging = self.root / ".staging" / uuid.uuid4().hex
            staging.mkdir(parents=True, exist_ok=False)
            try:
                for relative, value in changes.items():
                    destination = self._path(relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staged = staging / Path(*_safe_relative(relative).parts)
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    staged.write_bytes(_canonical_bytes(value))
                    json.loads(staged.read_text(encoding="utf-8"))
                new_revision = actual + 1
                commit = {"revision": new_revision, "run_id": run_id, "actor": actor,
                          "paths": sorted(changes), "dependencies": dependencies or {},
                          "based_on_revision": expected_revision}
                (staging / "commit.json").write_bytes(_canonical_bytes(commit))
                for relative in changes:
                    staged = staging / Path(*_safe_relative(relative).parts)
                    _atomic_replace(self._path(relative), staged.read_bytes())
                manifest["revision"] = new_revision
                manifest["status"] = self._status_for_paths(manifest.get("status", "empty"), changes)
                self._refresh_parts(manifest, revision=new_revision, changed_paths=changes)
                for relative in changes:
                    manifest["parts"][PurePosixPath(relative).as_posix()]["dependencies"] = dependencies or {}
                self._write_manifest(manifest)
                return BIRCommit(new_revision, digest_file(self.manifest_path), tuple(sorted(changes)), manifest["status"])
            finally:
                shutil.rmtree(staging, ignore_errors=True)

    def atomic_write(self, relative_path: str, value: dict[str, Any]) -> None:
        manifest = self._read_manifest()
        self.transaction(relative_path and {relative_path: value}, run_id="system",
                         actor="atomic_write", expected_revision=int(manifest["revision"]))

    def materialize(self, relative_path: str, destination: str | Path) -> str:
        source = self._path(relative_path)
        if not source.is_file():
            raise ValueError(f"BIR part does not exist: {relative_path}")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_replace(target, source.read_bytes())
        return str(target)

    def recover(self) -> None:
        staging_root = self.root / ".staging"
        if not staging_root.is_dir():
            return
        # Keep interrupted staging trees so a caller can inspect or resume a
        # transaction.  They are excluded from the canonical manifest and
        # discovery, and a later explicit cleanup can remove them safely.
        staging_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _status_for_paths(current: str, changes: dict[str, Any]) -> str:
        paths = set(changes)
        if "benchmark_info.json" in paths:
            return "specified"
        if any(path.endswith("/access.json") for path in paths):
            return "datasets_ready"
        if any("planning/" in path for path in paths):
            return "planning"
        if "allocation.json" in paths:
            return "allocated"
        if any(path.startswith("outputs/") for path in paths):
            return "generated"
        if "delivery.json" in paths:
            return "delivered"
        return current


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        path = Path(value)
        return path.is_absolute() or (len(value) > 1 and value[1] == ":" and value[0].isalpha())
    return False


def _atomic_replace(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

