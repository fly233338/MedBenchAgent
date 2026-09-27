"""Pinned Docker or local Python execution for the same preparation operations."""

import importlib.metadata
import json
import os
import shutil
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


class ExecutionUnavailable(RuntimeError):
    """An execution environment failure that agent argument repairs cannot resolve."""


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
    if saved and "requested_backend" not in saved:
        raise ValueError("Obsolete preparation runtime; start a new run")
    mode = saved["backend"] if saved else settings["backend"]
    reason = None
    if mode in {"auto", "docker"}:
        try:
            identity = docker_identity(settings, smoke=saved is None)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            if mode == "docker":
                raise ExecutionUnavailable(f"Pinned Docker runtime unavailable: {exc}") from exc
            reason = str(exc)
            identity = local_identity()
    else:
        identity = local_identity()
    if saved and {k: v for k, v in saved.items() if k not in {"selection_reason", "requested_backend"}} != identity:
        raise ValueError("Preparation runtime changed; start a new preparation execution")
    return {**identity, "requested_backend": saved["requested_backend"] if saved else settings["backend"],
            "selection_reason": saved.get("selection_reason") if saved else reason}


def require_script_runtime(runtime):
    if runtime["backend"] == "local" and runtime["requested_backend"] != "local":
        raise ValueError("Generated Python requires explicit backend=local; start a new run, or use Docker")


def output_digests(output):
    from medbenchagent.data.sources import digest_file
    return {p.relative_to(output).as_posix(): digest_file(p) for p in sorted(Path(output).rglob("*"))
            if p.is_file() and not {"scratch", "cache"}.intersection(p.relative_to(output).parts)}


def verify_output_digests(output, expected):
    from medbenchagent.data.sources import digest_file
    from medbenchagent.data.validation import source_image_path
    if not expected:
        raise ValueError("Prepared operation has no output binding")
    for name, digest in expected.items():
        if digest_file(source_image_path(output, name)) != digest:
            raise ValueError("Prepared operation output changed or is incomplete; use a new operation")


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
        raise ExecutionUnavailable("Preparation exceeded its execution budget") from None
    if process.returncode:
        hint = " Install local processing dependencies with uv sync --extra data-processing." if "ModuleNotFoundError" in stderr else ""
        if hint:
            raise ExecutionUnavailable((stdout + "\n" + stderr)[-5000:] + hint)
        raise ValueError((stdout + "\n" + stderr)[-5000:] + hint)


def run(context, output, *, script=None, request=None, annotation=None, specification=None, inputs=None, provenance=None):
    """Execute a fixed operation or generated script with an immutable completion receipt."""
    from medbenchagent.data.sources import verify_source, digest_file

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = get_settings().data_processing
    runtime_file = Path(context.get("bundle_root", output.parent)) / "execution_runtime.json"
    saved = json.loads(runtime_file.read_text(encoding="utf-8")) if runtime_file.is_file() else context.get("execution_runtime")
    runtime = select_backend(settings, saved)
    if script:
        require_script_runtime(runtime)
    atomic_write_json(runtime_file, runtime)
    context["execution_runtime"] = runtime
    if context.get("source_manifest"):
        verify_source(context["source_manifest"])
    kind = "script" if script else request["operation"]
    receipts = Path(context.get("operation_directory", output.parent)) / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / f"{output.name}-{kind}.complete.json"
    identity = {"runtime": runtime, "script": digest_file(script) if script else None,
                "request": request, "annotation": annotation,
                "annotation_digest": digest_file(annotation) if annotation else None,
                "specification": digest_file(specification) if specification else None,
                "inputs": digest_file(inputs) if inputs else None,
                "provenance": digest_file(provenance) if provenance else None,
                "dependencies": context.get("artifact_versions", {})}
    if receipt.is_file():
        saved_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        if saved_receipt["identity"] != identity:
            raise ValueError("Prepared operation dependencies changed; use a new operation directory")
        verify_output_digests(output, saved_receipt["outputs"])
        if (output / ("result.json" if request else "annotation.jsonl")).is_file():
            return json.loads((output / "result.json").read_text(encoding="utf-8")) if request else None
    elif any(output.iterdir()):
        incomplete = Path(context.get("operation_directory", output.parent)) / "incomplete"
        incomplete.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(incomplete / (output.name + "-" + uuid4().hex)))
        output.mkdir()
    request_path = receipts / (output.name + "-request.json")
    if request is not None:
        atomic_write_json(request_path, request)
    source = str(Path(context["source_root"]).resolve())
    project = str(Path(__file__).resolve().parents[2])
    scratch = Path(context.get("operation_directory", output.parent)) / "scratch" / output.name
    scratch.mkdir(parents=True, exist_ok=True)
    environment = {"MEDBENCH_DATA_MEMORY": settings["memory_limit"],
                   "MEDBENCH_DATA_THREADS": str(settings["threads"]),
                   "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    if runtime["backend"] == "local":
        environment.update(PYTHONPATH=project, MEDBENCH_SOURCE_ROOT=source,
                           MEDBENCH_WORK_DIR=str(output), MEDBENCH_DATA_SCRATCH=str(scratch),
                           MEDBENCH_ANNOTATION_PATH=str(Path(annotation).resolve()) if annotation else "")
        for name, path in (("SPEC", specification), ("INPUTS", inputs), ("PROVENANCE", provenance)):
            if path:
                environment[f"MEDBENCH_{name}_PATH"] = str(Path(path).resolve())
        arguments = [str(Path(script).resolve())] if script else ["-m", "medbenchagent.data.preparation_ops", str(request_path)]
        _local_run([sys.executable, *arguments], cwd=output, env={**os.environ, **environment},
                   timeout=settings["timeout_seconds"])
    else:
        container_name = "medbench-data-" + uuid4().hex
        command = ["docker", "run", "--name", container_name, "--rm", "--pull=never", "--network=none",
                   "--read-only", f"--memory={settings['container_memory']}", f"--cpus={settings['threads']}",
                   "--pids-limit=128", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                   "--tmpfs", "/tmp:rw,size=256m", "--workdir", "/work"]
        environment.update(PYTHONPATH="/project", MEDBENCH_SOURCE_ROOT="/source", MEDBENCH_WORK_DIR="/work",
                           MEDBENCH_DATA_SCRATCH="/scratch", MEDBENCH_ANNOTATION_PATH="/annotation/annotation.jsonl" if annotation else "")
        for key, value in environment.items():
            command += ["--env", f"{key}={value}"]
        mounts = [(source, "/source", True), (project, "/project", True),
                  (str(output), "/work", False), (str(scratch), "/scratch", False)]
        for name, path in (("spec", specification), ("inputs", inputs), ("provenance", provenance)):
            if path:
                command += ["--env", f"MEDBENCH_{name.upper()}_PATH=/{name}.json"]
                mounts.append((str(Path(path).resolve()), f"/{name}.json", True))
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
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False, timeout=30)
            raise ExecutionUnavailable("Preparation exceeded its execution budget") from None
        if completed.returncode:
            if completed.returncode in {125, 126, 127} or "ModuleNotFoundError" in completed.stderr:
                raise ExecutionUnavailable((completed.stdout + "\n" + completed.stderr)[-5000:])
            raise ValueError((completed.stdout + "\n" + completed.stderr)[-5000:])
    if context.get("source_manifest"):
        verify_source(context["source_manifest"])
    expected = output / ("result.json" if request else "annotation.jsonl")
    if not expected.is_file():
        raise ValueError(f"Preparation did not produce {expected.name}")
    atomic_write_json(receipt, {"identity": identity, "outputs": output_digests(output)})
    return json.loads(expected.read_text(encoding="utf-8")) if request else None
