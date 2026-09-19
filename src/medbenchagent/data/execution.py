"""Pinned Docker or local Python execution for the same preparation operations."""

import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


PACKAGES = ("duckdb", "pydantic", "pandas", "openpyxl", "xlrd", "Pillow", "numpy",
            "pydicom", "nibabel", "pynrrd", "SimpleITK", "pylibjpeg",
            "pylibjpeg-libjpeg", "pylibjpeg-openjpeg")


def local_identity():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"backend": "local", "python": sys.executable, "version": sys.version,
            "packages": versions}


def docker_identity(settings, *, smoke=True):
    commands = [["docker", "info", "--format={{.OSType}}"],
                ["docker", "image", "inspect", "--format={{.Id}}", settings["image"]]]
    image_id = None
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        if completed.returncode:
            raise ValueError(completed.stderr.strip() or "Docker preparation runtime unavailable")
        image_id = completed.stdout.strip()
    if smoke:
        completed = subprocess.run(
            ["docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
             image_id, "python", "-c", "pass"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if completed.returncode:
            raise ValueError(completed.stderr.strip() or "Docker cannot start the preparation image")
    return {"backend": "docker", "image_id": image_id}


def select_backend(settings, saved=None):
    """Resolve auto only once; a saved runtime never silently switches backend."""
    mode = saved["backend"] if saved else settings["backend"]
    reason = None
    if mode in {"auto", "docker"}:
        try:
            identity = docker_identity(settings, smoke=saved is None)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            if mode == "docker":
                raise ValueError(f"Pinned Docker runtime unavailable: {exc}") from exc
            reason = str(exc)
            identity = local_identity()
    else:
        identity = local_identity()
    if saved and {k: v for k, v in saved.items() if k != "selection_reason"} != identity:
        raise ValueError("Preparation runtime changed; start a new preparation execution")
    return {**identity, "selection_reason": saved.get("selection_reason") if saved else reason}


def _local_run(command, *, cwd, env, timeout):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8",
                               start_new_session=os.name != "nt")
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False, timeout=30)
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise ValueError("Preparation exceeded its execution budget") from None
    if process.returncode:
        hint = " Install local processing dependencies with uv sync --extra data-processing." if "ModuleNotFoundError" in stderr else ""
        raise ValueError((stdout + "\n" + stderr)[-5000:] + hint)


def run(context, output, *, script=None, request=None, annotation=None):
    """Execute a fixed operation or generated script with an immutable completion receipt."""
    from medbenchagent.data.sources import verify_source, digest_file

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = get_settings().data_processing
    runtime_file = Path(context.get("bundle_root", output.parent)) / "execution_runtime.json"
    saved = json.loads(runtime_file.read_text(encoding="utf-8")) if runtime_file.is_file() else context.get("execution_runtime")
    runtime = select_backend(settings, saved)
    atomic_write_json(runtime_file, runtime)
    context["execution_runtime"] = runtime
    if context.get("source_manifest"):
        verify_source(context["source_manifest"])
    kind = "script" if script else request["operation"]
    receipt = output.parent / f"{output.name}-{kind}.complete.json"
    identity = {"runtime": runtime, "script": digest_file(script) if script else None,
                "request": request, "annotation": annotation,
                "dependencies": context.get("artifact_versions", {})}
    if receipt.is_file():
        if json.loads(receipt.read_text(encoding="utf-8")) != identity:
            raise ValueError("Prepared operation dependencies changed; use a new operation directory")
        if (output / ("result.json" if request else "annotation.jsonl")).is_file():
            return json.loads((output / "result.json").read_text(encoding="utf-8")) if request else None
    request_path = output.parent / (output.name + "-request.json")
    if request is not None:
        atomic_write_json(request_path, request)
    source = str(Path(context["source_root"]).resolve())
    project = str(Path(__file__).resolve().parents[2])
    environment = {"MEDBENCH_DATA_MEMORY": settings["memory_limit"],
                   "MEDBENCH_DATA_THREADS": str(settings["threads"]),
                   "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    if runtime["backend"] == "local":
        environment.update(PYTHONPATH=project, MEDBENCH_SOURCE_ROOT=source,
                           MEDBENCH_WORK_DIR=str(output), MEDBENCH_DATA_SCRATCH=str(output / "scratch"),
                           MEDBENCH_ANNOTATION_PATH=str(Path(annotation).resolve()) if annotation else "")
        arguments = [str(Path(script).resolve())] if script else ["-m", "medbenchagent.data.preparation_ops", str(request_path)]
        _local_run([sys.executable, *arguments], cwd=output, env={**os.environ, **environment},
                   timeout=settings["timeout_seconds"])
    else:
        name = "medbench-data-" + uuid4().hex
        command = ["docker", "run", "--name", name, "--rm", "--pull=never", "--network=none",
                   "--read-only", f"--memory={settings['container_memory']}", f"--cpus={settings['threads']}",
                   "--pids-limit=128", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--tmpfs", "/tmp:rw,size=256m", "--workdir", "/work"]
        environment.update(PYTHONPATH="/project", MEDBENCH_SOURCE_ROOT="/source", MEDBENCH_WORK_DIR="/work",
                           MEDBENCH_DATA_SCRATCH="/work/scratch", MEDBENCH_ANNOTATION_PATH="/annotation/annotation.jsonl" if annotation else "")
        for key, value in environment.items():
            command += ["--env", f"{key}={value}"]
        mounts = [(source, "/source", True), (project, "/project", True), (str(output), "/work", False)]
        if annotation:
            mounts.append((str(Path(annotation).resolve()), "/annotation/annotation.jsonl", True))
        mounts.append((str(Path(script).resolve()) if script else str(request_path), "/script.py" if script else "/request.json", True))
        for host, target, readonly in mounts:
            command += ["--mount", f"type=bind,source={host},target={target}" + (",readonly" if readonly else "")]
        arguments = ["python", "/script.py"] if script else ["python", "-m", "medbenchagent.data.preparation_ops", "/request.json"]
        try:
            completed = subprocess.run(command + [runtime["image_id"]] + arguments, capture_output=True,
                                       text=True, encoding="utf-8", timeout=settings["timeout_seconds"], check=False)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False, timeout=30)
            raise ValueError("Preparation exceeded its execution budget") from None
        if completed.returncode:
            raise ValueError((completed.stdout + "\n" + completed.stderr)[-5000:])
    if context.get("source_manifest"):
        verify_source(context["source_manifest"])
    expected = output / ("result.json" if request else "annotation.jsonl")
    if not expected.is_file():
        raise ValueError(f"Preparation did not produce {expected.name}")
    atomic_write_json(receipt, identity)
    return json.loads(expected.read_text(encoding="utf-8")) if request else None
