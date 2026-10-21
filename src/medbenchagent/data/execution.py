"""Backend selection and dependency identity for preparation runtimes."""

import importlib.metadata
import subprocess
import sys



PACKAGES = ("duckdb", "pydantic", "pandas", "openpyxl", "xlrd", "Pillow", "numpy",
            "pydicom", "nibabel", "pynrrd", "SimpleITK", "pylibjpeg",
            "pylibjpeg-libjpeg", "pylibjpeg-openjpeg")


class ExecutionUnavailable(RuntimeError):
    """An execution environment failure that agent argument repairs cannot resolve."""


def data_dependency_identity():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def local_identity():
    return {"backend": "local", "python": sys.executable, "version": sys.version}


def docker_identity():
    """Check the shared Docker engine; component images are checked at use."""
    completed = subprocess.run(["docker", "info", "--format={{.OSType}}"],
                               capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode or getattr(completed, "stdout", "").strip() != "linux":
        raise ValueError(getattr(completed, "stderr", "").strip() or "A Linux Docker daemon is required")
    version = subprocess.run(["docker", "version", "--format={{.Server.Version}}"],
                             capture_output=True, text=True, timeout=30, check=False)
    if version.returncode or not getattr(version, "stdout", "").strip():
        raise ValueError(getattr(version, "stderr", "").strip() or "Docker server version is unavailable")
    return {"backend": "docker", "docker_server_version": version.stdout.strip()}


def docker_image_id(image: str) -> str:
    """Resolve one component image without choosing a backend or pulling it."""
    try:
        completed = subprocess.run(["docker", "image", "inspect", "--format={{.Id}}", image],
                                   capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExecutionUnavailable(f"Docker image unavailable: {image}") from exc
    if completed.returncode or not getattr(completed, "stdout", "").strip():
        raise ExecutionUnavailable(f"Docker image unavailable: {image}")
    return completed.stdout.strip()


def select_backend(settings, saved=None):
    """Resolve auto only once; a saved runtime never silently switches backend."""
    if saved:
        if "requested_backend" not in saved:
            raise ValueError("Obsolete execution runtime; start a new run")
        current = (local_identity() if saved["backend"] == "local"
                   else docker_identity())
        if any(saved.get(key) != value for key, value in current.items()):
            raise ValueError("Preparation runtime changed; start a new run")
        return saved
    mode = saved["backend"] if saved else settings["backend"]
    reason = None
    if mode in {"auto", "docker"}:
        try:
            identity = docker_identity()
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            if mode == "docker":
                raise ExecutionUnavailable(f"Pinned Docker runtime unavailable: {exc}") from exc
            reason = str(exc)
            identity = local_identity()
    else:
        identity = local_identity()
    return {**identity, "requested_backend": settings["backend"], "selection_reason": reason}
