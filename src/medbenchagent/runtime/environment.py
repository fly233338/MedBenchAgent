"""Prepare local runtime dependencies before starting a user-facing command."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from filelock import FileLock

from medbenchagent.data.execution import data_dependency_identity, select_backend
from medbenchagent.settings import PROJECT_ROOT, AppSettings, load_settings


_WORKER_REQUIREMENTS = PROJECT_ROOT / "config" / "coding-worker.requirements.txt"
_WORKER_MARKER = ".medbench-requirements.sha256"


def _uv() -> str:
    executable = shutil.which("uv")
    if executable:
        return executable
    raise RuntimeError(
        "Local data processing needs uv to prepare dependencies. "
        "Install uv or select data_processing.backend = 'docker'."
    )


def _run_uv(arguments: list[str], *, timeout: int = 1800) -> None:
    try:
        completed = subprocess.run(
            [_uv(), *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Dependency setup timed out: uv {' '.join(arguments)}") from exc
    if completed.returncode:
        details = (completed.stderr or completed.stdout or "").strip()
        if len(details) > 2000:
            details = details[-2000:]
        raise RuntimeError(
            f"Dependency setup failed: uv {' '.join(arguments)}\n{details}"
        )


def _requirements_digest() -> str:
    return hashlib.sha256(_WORKER_REQUIREMENTS.read_bytes()).hexdigest()


def _worker_is_ready(executable: Path, digest: str, marker: Path) -> bool:
    if not executable.is_file() or not marker.is_file():
        return False
    return marker.read_text(encoding="utf-8").strip() == digest


def _ensure_worker(settings: AppSettings) -> None:
    worker_root = Path(settings.data_processing["coding_worker_venv"])
    executable = worker_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = worker_root / _WORKER_MARKER
    digest = _requirements_digest()
    if _worker_is_ready(executable, digest, marker):
        return

    _run_uv(["venv", "--allow-existing", str(worker_root)])
    _run_uv(["pip", "install", "--python", str(executable), "-r", str(_WORKER_REQUIREMENTS)])
    marker.write_text(digest + "\n", encoding="utf-8")


def _ensure_data_dependencies() -> None:
    missing = [name for name, version in data_dependency_identity().items() if version is None]
    if not missing:
        return
    _run_uv(["sync", "--extra", "data-processing"])
    remaining = [name for name, version in data_dependency_identity().items() if version is None]
    if remaining:
        raise RuntimeError(
            "Local data processing dependencies are still missing after setup: "
            + ", ".join(remaining)
        )


def ensure_environment(config_path: Path | None = None) -> dict[str, object]:
    """Ensure local data-processing dependencies before entering the UI/CLI.

    Docker-backed runs use their pinned images and do not need local setup. A
    lock makes concurrent CLI launches share one idempotent setup operation.
    """
    settings = load_settings(config_path)
    requested = settings.data_processing["backend"]
    if requested == "docker":
        return {"backend": "docker", "prepared": False}

    runtime = select_backend(settings.data_processing)
    if runtime["backend"] != "local":
        return {"backend": runtime["backend"], "prepared": False}

    worker_root = Path(settings.data_processing["coding_worker_venv"])
    worker_python = worker_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    worker_ready = _worker_is_ready(
        worker_python,
        _requirements_digest(),
        worker_root / _WORKER_MARKER,
    )
    data_missing = any(version is None for version in data_dependency_identity().values())
    if data_missing or not worker_ready:
        print("Preparing local data-processing environment...", file=sys.stderr, flush=True)

    lock_path = PROJECT_ROOT / ".runtime" / "environment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        _ensure_data_dependencies()
        _ensure_worker(settings)
    return {"backend": "local", "prepared": True}


__all__ = ["ensure_environment"]
