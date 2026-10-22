"""File handoffs, not a versioned artifact store or an operation ledger."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .contracts import ArtifactRef, WorkRequest


class ArtifactIntegrityError(ValueError):
    """The registered file no longer has its accepted content identity."""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _within(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = Path(relative)
    if path.is_absolute() or path.drive or not relative:
        raise ValueError(f"Artifact paths must be relative: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Artifact escapes its root: {relative}")
    return resolved


def resolve_artifact(root: str | Path, reference: ArtifactRef) -> Path:
    path = _within(Path(root), reference.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if _digest(path) != reference.digest:
        raise ArtifactIntegrityError(f"Artifact content changed: {reference.artifact_id}")
    return path


def register_file(
    root: str | Path, request: WorkRequest, path: str, *, kind: str
) -> ArtifactRef:
    """Register an existing file owned by this request; never accept a claimed file."""
    root = Path(root).resolve()
    actual = _within(root, path)
    owned = _within(root, f"artifacts/{request.request_id}")
    if not actual.is_relative_to(owned) or not actual.is_file():
        raise ValueError(f"Expected a file in this request's output directory: {path}")
    relative = actual.relative_to(root).as_posix()
    identifier = uuid.uuid5(
        uuid.NAMESPACE_URL, f"{request.compilation_id}/{request.request_id}/{relative}"
    ).hex
    return ArtifactRef(artifact_id=identifier, kind=kind, path=relative, digest=_digest(actual))


def write_artifact(
    root: str | Path, request: WorkRequest, name: str, *, kind: str, content: bytes
) -> ArtifactRef:
    """Write once; replay may reuse identical bytes, but cannot revise a handoff."""
    root = Path(root).resolve()
    owned = _within(root, f"artifacts/{request.request_id}")
    path = _within(owned, name)
    if not isinstance(content, bytes):
        raise TypeError("Artifact content must be bytes")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Cannot overwrite an existing artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _within(owned, str(path.relative_to(owned)) + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    return register_file(root, request, path.relative_to(root).as_posix(), kind=kind)


def write_json_artifact(
    root: str | Path, request: WorkRequest, name: str, *, kind: str, value: Any
) -> ArtifactRef:
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode("utf-8")
    return write_artifact(root, request, name, kind=kind, content=content)


def read_json_artifact(root: str | Path, reference: ArtifactRef) -> Any:
    return json.loads(resolve_artifact(root, reference).read_text(encoding="utf-8"))


def preview_artifact(root: str | Path, reference: ArtifactRef, *, limit: int) -> dict[str, Any]:
    """Bounded text only; no recursive dataset or workspace exploration."""
    if limit <= 0:
        raise ValueError("Artifact preview limit must be positive")
    preview = {"artifact_id": reference.artifact_id, "kind": reference.kind}
    try:
        with resolve_artifact(root, reference).open(encoding="utf-8") as stream:
            content = stream.read(limit + 1)
        preview.update(content=content[:limit], truncated=len(content) > limit)
    except UnicodeError:
        preview["unavailable"] = "Artifact is not UTF-8 text"
    except ArtifactIntegrityError as exc:
        preview["unavailable"] = str(exc)
        preview["issue_code"] = "artifact_integrity"
    except OSError:
        preview["unavailable"] = "Registered artifact is not readable"
    return preview
