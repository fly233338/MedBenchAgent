"""Canonical preparation workspace and path mapping.

The worker only receives paths produced by this module.  Keeping layout
construction in one place prevents local and Docker runs from silently using
different source roots or writing into a source/bundle directory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Stage, digest_file


_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")


def validate_identifier(value: str, *, field: str = "identifier") -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must contain only letters, digits, '.', '_' and '-': {value!r}")
    return value


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class WorkspaceLayout:
    """All host and worker paths for one stage attempt."""

    run_root: Path
    attempt_root: Path
    stage: Stage
    attempt_id: str
    session_seq: int
    source_roots: dict[str, Path]
    backend: str = "local"

    @classmethod
    def create(
        cls,
        run_root: str | Path,
        *,
        dataset_id: str,
        stage: Stage | str,
        attempt_id: str,
        session_seq: int = 0,
        source_roots: dict[str, str | Path] | None = None,
        backend: str = "local",
    ) -> "WorkspaceLayout":
        stage_value = Stage(stage)
        validate_identifier(dataset_id, field="dataset_id")
        validate_identifier(attempt_id, field="attempt_id")
        if session_seq < 0:
            raise ValueError("session_seq must be non-negative")
        root = _resolved(run_root)
        attempt = root / "preparation" / dataset_id / "attempts" / stage_value.value / attempt_id
        sources = {
            validate_identifier(str(source_id), field="source_id"): _resolved(source)
            for source_id, source in (source_roots or {}).items()
        }
        layout = cls(root, attempt, stage_value, attempt_id, session_seq, sources, backend)
        layout.ensure()
        layout.validate_isolation()
        return layout

    @property
    def inputs(self) -> Path:
        return self.attempt_root / "inputs"

    @property
    def accepted(self) -> Path:
        return self.inputs / "accepted"

    @property
    def previous(self) -> Path:
        return self.inputs / "previous"

    @property
    def work(self) -> Path:
        return self.attempt_root / "work"

    @property
    def outputs(self) -> Path:
        return self.attempt_root / "outputs"

    @property
    def state(self) -> Path:
        return self.attempt_root / "state"

    @property
    def logs(self) -> Path:
        return self.attempt_root / "logs"

    @property
    def conversations(self) -> Path:
        return self.attempt_root / "conversations"

    @property
    def context_path(self) -> Path:
        return self.inputs / "context.json"

    @property
    def prompt_path(self) -> Path:
        return self.inputs / "prompt.md"

    @property
    def resume_state_path(self) -> Path:
        return self.state / "resume.json"

    @property
    def request_path(self) -> Path:
        return self.state / "request.json"

    @property
    def result_path(self) -> Path:
        return self.state / "result.json"

    @property
    def events_path(self) -> Path:
        return self.logs / "coding_events.jsonl"

    def ensure(self) -> None:
        for directory in (self.inputs, self.accepted, self.work, self.outputs, self.state, self.logs, self.conversations):
            directory.mkdir(parents=True, exist_ok=True)
        identity = self.state / "identity.json"
        payload = {
            "stage": self.stage.value,
            "attempt_id": self.attempt_id,
            "session_seq": self.session_seq,
            "backend": self.backend,
            "sources": {key: str(value) for key, value in sorted(self.source_roots.items())},
        }
        if identity.is_file():
            existing = json.loads(identity.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError(f"Attempt directory identity changed: {self.attempt_root}")
        else:
            identity.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")

    def validate_isolation(self) -> None:
        attempt = _resolved(self.attempt_root)
        if _inside(attempt, self.run_root.parent) is False:
            raise ValueError("Attempt workspace must remain below the run root")
        for source_id, source in self.source_roots.items():
            if _inside(attempt, source) or _inside(source, attempt):
                raise ValueError(f"Source and attempt workspace overlap: {source_id}")
        for output in (self.outputs, self.work, self.state):
            if not _inside(_resolved(output), attempt):
                raise ValueError(f"Workspace path escapes attempt root: {output}")

    def source_bindings(self) -> list[dict[str, Any]]:
        """Return stable host/worker mappings for context and input configs."""
        bindings = []
        for source_id, source in sorted(self.source_roots.items()):
            worker_root = f"/workspace/sources/{source_id}" if self.backend == "docker" else str(source)
            bindings.append({
                "source_id": source_id,
                "host_root": str(source),
                "worker_root": worker_root,
                "read_only": True,
                "digest": self.source_digest(source),
            })
        return bindings

    @staticmethod
    def source_digest(source: Path) -> str:
        if not source.exists():
            raise ValueError(f"Source path is missing: {source}")
        if source.is_file():
            return digest_file(source)
        # A manifest is the authoritative source identity.  For a directory
        # fallback use relative names and bytes without embedding host paths.
        import hashlib
        hasher = hashlib.sha256()
        for item in sorted(path for path in source.rglob("*") if path.is_file()):
            hasher.update(item.relative_to(source).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            with item.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
            hasher.update(b"\0")
        return hasher.hexdigest()

    def command(self, script: str, *, input_config: str, output_dir: str, mode: str) -> list[str]:
        if Path(script).is_absolute() or ".." in Path(script).parts:
            raise ValueError(f"Script must be under outputs: {script}")
        return ["python", f"outputs/{Path(script).as_posix()}", "--input-config", input_config, "--output-dir", output_dir, "--mode", mode]

    def write_config(self, name: str, payload: dict[str, Any]) -> Path:
        if Path(name).name != name or not name.endswith("_config.json"):
            raise ValueError("Workspace configs must be simple *_config.json names")
        path = self.inputs / name
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        return path

    def mark_inputs_readonly(self) -> None:
        """Best effort input write protection for local workers.

        Docker mounts enforce read-only access.  On Windows chmod is advisory,
        so the runtime also validates source digests before and after a run.
        """
        for path in (self.inputs, self.accepted):
            for item in [path, *path.rglob("*")]:
                try:
                    mode = item.stat().st_mode
                    item.chmod(mode & ~0o222)
                except OSError:
                    pass


def validate_path_contract(path: str | Path, *, root: str | Path, allow_absolute: bool = False) -> Path:
    value = Path(path)
    if value.is_absolute() and not allow_absolute:
        raise ValueError(f"Artifact path must be relative: {path}")
    resolved = _resolved(value if value.is_absolute() else Path(root) / value)
    if not _inside(resolved, _resolved(root)):
        raise ValueError(f"Artifact path escapes workspace: {path}")
    return resolved
