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


class BIRStaleError(ValueError):
    """Valid content whose producing business inputs have changed."""


@dataclass(frozen=True)
class BIRCommit:
    revision: int
    manifest_digest: str
    changed_paths: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class BIRValidationReport:
    valid: bool
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


def business_dependencies(context, *paths, registry=False):
    store = BIRStore(context["bir_path"])
    dependencies = {path: store.digest(path) for path in paths}
    if registry:
        dependencies["registry"] = store.registry().digest
    if config := context.get("effective_config_path"):
        dependencies["file:" + str(Path(config).resolve())] = digest_file(config)
    return dependencies


def business_snapshot(context):
    store = BIRStore(context["bir_path"])
    return {p.relative_to(store.root).as_posix(): digest_file(p) for p in store.root.rglob("*.json")
            if not p.is_relative_to(store.root / "registry")}


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

    def prompt(self, name: str, *, include_shared: bool = True) -> str:
        try:
            prompt = self.prompts[name]
            if include_shared and name in {"task_planning", "axis_analysis", "scenario_screening", "leaf_task_design", "scenario_review"}:
                return self.prompts["shared_planning_policy"] + "\n\n" + prompt
            return prompt
        except KeyError as exc:
            raise ValueError(f"BIR prompt is missing: {name}") from exc

    def planning_extensions(self) -> PlanningExtensionCatalog:
        return self.extensions

    def validate_value(self, axis: str, value: str) -> None:
        if value not in self.axis_values(axis):
            raise ValueError(f"Unknown value {value!r} for BIR Axis {axis}")


class BIRStore:
    """Safe, revisioned access to one compilation's BIR directory."""

    SCHEMA_VERSION = "bir"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.lock = FileLock(str(self.root / ".bir.lock"))
        if not self.root.is_dir():
            raise ValueError(f"BIR directory does not exist: {self.root}")

    @classmethod
    def load_template(cls, template_dir: str | Path) -> BIRTemplate:
        path = Path(template_dir).resolve()
        if not path.is_dir():
            raise ValueError(f"BIR template directory does not exist: {path}")
        for required in ("manifest.json", "benchmark_info.json", "allocation.json", "delivery.json", "planning_extensions.json"):
            if not (path / required).is_file():
                raise ValueError(f"BIR template requires {required}")
        RegistrySnapshot.from_directory(path / "registry")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("BIR template requires manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        template_revision = manifest["template_revision"]
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
    def initialize(cls, target_dir: str | Path, template_dir: str | Path, compilation_id: str) -> "BIRStore":
        target = Path(target_dir).resolve()
        if target.exists():
            raise ValueError(f"BIR directory already exists: {target}")
        template = cls.load_template(template_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(template.path, target)
        store = cls.__new__(cls)
        store.root = target
        store.lock = FileLock(str(target / ".bir.lock"))
        manifest = {
            "schema_version": cls.SCHEMA_VERSION,
            "compilation_id": compilation_id,
            "template_revision": template.template_revision,
            "template_digest": template.digest,
            "registry_digest": RegistrySnapshot.from_directory(target / "registry").digest,
        }
        store._write_manifest(manifest)
        for part in target.rglob("*.json"):
            if part.name != "manifest.json" and not part.is_relative_to(target / "registry"):
                payload = json.loads(part.read_text(encoding="utf-8"))
                payload["_commit"] = {"revision": 0, "digest": _digest_bytes(_canonical_bytes(payload)), "dependencies": {}}
                _atomic_replace(part, _canonical_bytes(payload))
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

    def load(self, relative_path, *, expected_schema=None, verify_digest=True):
        path = self._path(relative_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not relative_path.startswith("registry/") and relative_path != "manifest.json":
            commit = value.pop("_commit", None)
            if commit is None:
                raise ValueError(f"BIR document has no atomic commit metadata: {relative_path}")
            if verify_digest:
                if commit["digest"] != _digest_bytes(_canonical_bytes(value)):
                    raise ValueError(f"BIR content digest mismatch: {relative_path}")
                for dependency, expected in commit["dependencies"].items():
                    if dependency == "registry":
                        actual = self.registry().digest
                    elif dependency.startswith("file:"):
                        actual = digest_file(dependency[5:])
                    else:
                        self.load(dependency)
                        actual = self.digest(dependency)
                    if actual != expected:
                        raise BIRStaleError(f"BIR dependency changed: {dependency}; rebuild {relative_path}")
        if expected_schema and value.get("schema_version") != expected_schema:
            raise ValueError(f"Unexpected BIR schema: {relative_path}")
        return value

    def current(self, relative_path):
        """Read a committed current business document, or an absent/stale stage."""
        try:
            value = self.load(relative_path)
            commit = json.loads(self._path(relative_path).read_text(encoding="utf-8")).get("_commit")
            return value if commit is None or commit["revision"] else None
        except (FileNotFoundError, BIRStaleError):
            return None

    def registry(self):
        snapshot = RegistrySnapshot.from_directory(self.root / "registry")
        if snapshot.digest != self._read_manifest()["registry_digest"]:
            raise ValueError("Frozen BIR Registry changed")
        return snapshot

    def digest(self, relative_path):
        return digest_file(self._path(relative_path))

    def validate(self):
        errors = []
        for path in self.root.rglob("*.json"):
            if path.is_relative_to(self.root / "registry"):
                continue
            try:
                self.load(path.relative_to(self.root).as_posix())
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        try:
            self.registry()
        except ValueError as exc:
            errors.append(str(exc))
        return BIRValidationReport(not errors, tuple(errors))

    def update(self, relative_path, value, *, run_id, actor, expected_revision=None,
               dependencies=None, expected_digest=None):
        if relative_path == "dataset_binding.json":
            from medbenchagent.data.contracts import DatasetAccessReceipt
            DatasetAccessReceipt.model_validate(value["access"]).verify()
        if relative_path.startswith("registry/") or relative_path == "manifest.json":
            raise ValueError("BIR Registry and experiment identity are immutable")
        with self.lock:
            path = self._path(relative_path)
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            previous_commit = previous.pop("_commit", {})
            revision = previous_commit.get("revision", 0)
            if expected_revision is not None and revision != expected_revision:
                raise BIRConflictError("BIR revision conflict")
            if expected_digest and (not path.exists() or digest_file(path) != expected_digest):
                raise BIRConflictError("BIR digest conflict")
            payload = dict(value)
            payload.pop("_commit", None)
            deps = dependencies or {}
            if previous == payload and previous_commit.get("dependencies") == deps:
                return BIRCommit(revision, digest_file(self.manifest_path), (), "unchanged")
            # Validate dependencies before the single visible state transition.
            for dependency, expected in deps.items():
                if dependency == "registry":
                    actual = self.registry().digest
                elif dependency.startswith("file:"):
                    actual = digest_file(dependency[5:])
                else:
                    if dependency == relative_path:
                        raise ValueError("BIR document cannot depend on itself")
                    self.load(dependency)
                    actual = self.digest(dependency)
                if actual != expected:
                    raise BIRConflictError(f"Dependency changed before commit: {dependency}")
            commit = {"revision": revision + 1, "digest": _digest_bytes(_canonical_bytes(payload)),
                      "dependencies": deps, "run_id": run_id, "actor": actor}
            _atomic_replace(path, _canonical_bytes({**payload, "_commit": commit}))
            return BIRCommit(revision + 1, digest_file(self.manifest_path), (relative_path,), "committed")

    def atomic_write(self, relative_path, value):
        return self.update(relative_path, value, run_id="system", actor="atomic_write")

    def materialize(self, relative_path, destination):
        # Explicit human-readable export, never a restore prerequisite.
        _atomic_replace(Path(destination), _canonical_bytes(self.load(relative_path)))
        return str(destination)


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



def persist_task_plan(context, plan):
    store = BIRStore(context["bir_path"])
    store.update("task_plan.json", plan.model_dump(mode="json"), run_id=context["run_id"], actor="task_planning",
        dependencies=business_dependencies(context, "benchmark_info.json", "dataset_binding.json", "planning_extensions.json", registry=True),
        expected_digest=context.get("planning_read_digest"))
    context["planning_read_digest"] = store.digest("task_plan.json")
    return {"task_plan_path": str(store.root / "task_plan.json")}
