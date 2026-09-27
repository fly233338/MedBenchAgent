"""A narrow workspace worker boundary for data preparation tasks."""

from __future__ import annotations

import os
import filecmp
import json
import shutil
import signal
import subprocess
import uuid
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.data.sources import verify_source
from medbenchagent.runtime.run_artifacts import atomic_write_json


class CodingTask(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    task_id: str
    kind: Literal["source_inventory", "dataset_adapter"]
    source_root: str
    source_manifest: str
    workspace: str
    objective: str
    required_outputs: list[str]
    inputs: dict[str, str] = Field(default_factory=dict)
    feedback: str = ""
    max_iterations: int = Field(default=40, ge=1)
    timeout_seconds: int = Field(default=1800, ge=1)
    conversation_id: str | None = None
    model: str
    base_url: str
    construction_events_path: str = ""
    run_id: str = ""
    dataset_id: str = ""
    source_id: str = ""


class CodingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["complete", "needs_input", "blocked", "failed"]
    conversation_id: str
    outputs: dict[str, str] = Field(default_factory=dict)
    question: str = ""
    diagnostic: str = ""


class CodingWorker(Protocol):
    def run(self, task: CodingTask) -> CodingResult: ...


class OpenHandsCodingWorker:
    """Run the OpenHands SDK in its Python 3.12 environment, outside this project venv."""

    _requirements = ("openhands-sdk==1.49.6", "openhands-tools==1.49.6",
                     "pandas==2.3.3", "openpyxl==3.1.5", "xlrd==2.0.2",
                     "duckdb==1.5.4", "Pillow==12.3.0")

    def __init__(self, *, uv: str = "uv"):
        self.uv = uv

    def run(self, task: CodingTask) -> CodingResult:
        verify_source(task.source_manifest)
        workspace = Path(task.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        source_copy = workspace / "source"
        if source_copy.is_symlink():
            raise ValueError("Worker source snapshot cannot be a symlink")
        if not source_copy.exists():
            # The worker sees a disposable source snapshot. Only the validator reads the original.
            shutil.copytree(task.source_root, source_copy)
        else:
            for staged in source_copy.rglob("*"):
                if staged.is_symlink():
                    staged.unlink()
            listed = set()
            with (Path(task.source_manifest) / "files.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    relative = json.loads(line)["path"]
                    listed.add(relative)
                    original = Path(task.source_root) / relative
                    staged = source_copy / relative
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    if not staged.is_file() or not filecmp.cmp(original, staged, shallow=False):
                        if staged.exists():
                            staged.chmod(0o644)
                        shutil.copy2(original, staged)
            for staged in source_copy.rglob("*"):
                if staged.is_file() and staged.relative_to(source_copy).as_posix() not in listed:
                    staged.chmod(0o644)
                    staged.unlink()
        for staged in source_copy.rglob("*"):
            if staged.is_file():
                staged.chmod(0o444)
        for name, path in task.inputs.items():
            destination = workspace / "inputs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        conversation_id = task.conversation_id or str(uuid.uuid5(uuid.NAMESPACE_URL, task.task_id))
        if task.conversation_id is None and not (workspace / "conversations" / conversation_id).exists():
            for name in task.required_outputs:
                (workspace / name).unlink(missing_ok=True)
        request = task.model_dump(exclude={"source_manifest", "inputs"})
        request["source_root"] = str(source_copy)
        request["conversation_id"] = conversation_id
        request_path = workspace / "coding_task.json"
        result_path = workspace / f"coding_result_{task.kind}.json"
        replay_marker = workspace / "source_inventory_replayed.json"
        replay_ok = (task.kind != "source_inventory" or
                     (replay_marker.is_file() and
                      json.loads(replay_marker.read_text(encoding="utf-8")).get("task_id") == task.task_id))
        if request_path.is_file():
            previous = json.loads(request_path.read_text(encoding="utf-8"))
            if previous == request:
                if result_path.is_file():
                    cached = CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
                    if cached.status == "complete" and all(Path(path).is_file() for path in cached.outputs.values()):
                        return self._validate_outputs(task, cached, workspace, conversation_id)
                elif (workspace / "conversations" / conversation_id).exists() and replay_ok and all(
                        (workspace / name).is_file() for name in task.required_outputs):
                    recovered = CodingResult(status="complete", conversation_id=conversation_id,
                                             outputs={name: str(workspace / name) for name in task.required_outputs})
                    return self._validate_outputs(task, recovered, workspace, conversation_id)
        result_path.unlink(missing_ok=True)
        atomic_write_json(request_path, request)
        runner = Path(__file__).resolve().parents[1] / "runtime" / "openhands_worker.py"
        command = [self.uv, "run", "--no-project", "--python", "3.12"]
        source_paths = []
        with (Path(task.source_manifest) / "files.jsonl").open(encoding="utf-8") as handle:
            source_paths = [json.loads(line)["path"].lower() for line in handle]
        extras = []
        if any(path.endswith(".dcm") for path in source_paths):
            extras.extend(("pydicom==3.0.1", "pylibjpeg==2.1.0",
                           "pylibjpeg-libjpeg==2.3.0", "pylibjpeg-openjpeg==2.5.0"))
        if any(path.endswith((".nii", ".nii.gz")) for path in source_paths):
            extras.append("nibabel==5.3.2")
        if any(path.endswith((".nrrd", ".nhdr")) for path in source_paths):
            extras.append("pynrrd==1.1.3")
        if any(path.endswith((".mha", ".mhd")) for path in source_paths):
            extras.append("SimpleITK==2.5.3")
        for package in (*self._requirements, *extras):
            command.extend(["--with", package])
        command.extend(["python", str(runner), str(request_path), str(result_path)])
        environment = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1"}
        try:
            with (workspace / "worker.stdout.log").open("a", encoding="utf-8") as stdout, \
                    (workspace / "worker.stderr.log").open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=workspace, env=environment,
                                           stdout=stdout, stderr=stderr,
                                           start_new_session=os.name != "nt",
                                           creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                                                          if os.name == "nt" else 0))
                try:
                    process.wait(timeout=task.timeout_seconds)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, check=False, timeout=30)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return CodingResult(status="failed", conversation_id=conversation_id,
                                        diagnostic="OpenHands worker exceeded its time budget")
        except OSError as exc:
            return CodingResult(status="failed", conversation_id=conversation_id,
                                diagnostic=f"OpenHands worker unavailable: {exc}")
        if process.returncode or not result_path.is_file():
            stderr = workspace / "worker.stderr.log"
            stdout = workspace / "worker.stdout.log"
            log = stderr if stderr.stat().st_size else stdout
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 4000))
                diagnostic = handle.read().decode("utf-8", errors="replace")
            return CodingResult(status="failed", conversation_id=conversation_id,
                                diagnostic=diagnostic)
        result = CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        return self._validate_outputs(task, result, workspace, conversation_id)

    @staticmethod
    def _validate_outputs(task, result, workspace, conversation_id):
        if result.conversation_id != conversation_id:
            raise ValueError("Worker returned another conversation identity")
        if result.status == "complete" and set(result.outputs) != set(task.required_outputs):
            raise ValueError("Worker did not return every required output")
        for name, path in result.outputs.items():
            resolved = Path(path).resolve()
            if (name not in task.required_outputs or not resolved.is_file()
                    or not resolved.is_relative_to(workspace)
                    or resolved != (workspace / name).resolve()):
                raise ValueError(f"Worker output is outside its contract: {name}")
        verify_source(task.source_manifest)
        return result


def coding_worker(_context) -> CodingWorker:
    return OpenHandsCodingWorker()
