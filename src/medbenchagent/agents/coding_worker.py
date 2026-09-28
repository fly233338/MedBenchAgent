"""OpenHands candidate work in Docker with a read-only source mount."""

from __future__ import annotations

import json
import filecmp
import os
import re
import shutil
import signal
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from medbenchagent.data.sources import verify_source
from medbenchagent.data.execution import ExecutionUnavailable, docker_image_id
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_llm_api_key, get_settings


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
    attempt: int = Field(default=1, ge=1)
    max_iterations: int = Field(default=40, ge=1)
    timeout_seconds: int = Field(default=1800, ge=1)
    conversation_id: str | None = None
    model: str
    base_url: str
    reasoning_effort: str = "medium"
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
    error_category: Literal["", "environment_unavailable", "model_unavailable",
                            "worker_execution_failed", "worker_output_invalid",
                            "source_changed", "source_semantics_unresolved"] = ""
    retryable: bool = False


class CodingWorker(Protocol):
    def run(self, task: CodingTask) -> CodingResult: ...


def _container_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return base_url
    host = "host.docker.internal"
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
    if "," in str(source):
        raise ValueError("Docker mount source path cannot contain a comma")
    return f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")


class OpenHandsCodingWorker:
    """Run candidate coding on the backend frozen for the parent run."""

    _requirements = (
        "openhands-sdk==1.49.6", "openhands-tools==1.49.6", "pandas==2.3.3",
        "openpyxl==3.1.5", "xlrd==2.0.2", "duckdb==1.5.4", "Pillow==12.3.0",
        "pydicom==3.0.1", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg==2.3.0",
        "pylibjpeg-openjpeg==2.5.0", "nibabel==5.3.2", "pynrrd==1.1.3",
        "SimpleITK==2.5.3",
    )

    def __init__(self, runtime=None):
        self.runtime = runtime

    def run(self, task: CodingTask) -> CodingResult:
        verify_source(task.source_manifest)
        settings = get_settings().data_processing
        # Production callers pass the run-frozen resolver result. Direct worker use
        # defaults to the Docker component so it never performs a second backend choice.
        runtime = self.runtime or {"backend": "docker", "requested_backend": "docker"}
        workspace = Path(task.workspace).resolve()
        source_root = Path(task.source_root).resolve()
        if workspace.is_relative_to(source_root) or source_root.is_relative_to(workspace):
            raise ValueError("Worker workspace and original source must be separate")
        for name in ("inputs", "work", "outputs", "state", "logs", "conversations"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        self._forward_events(workspace, task.construction_events_path)
        source = source_root
        if runtime["backend"] == "local" or settings["source_workspace_mode"] == "copy":
            source = workspace / ("source" if runtime["backend"] == "local" else "source_snapshot")
            if not source.exists():
                shutil.copytree(source_root, source)
            listed = set()
            with (Path(task.source_manifest) / "files.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    relative = json.loads(line)["path"]
                    listed.add(relative)
                    staged = source / relative
                    if staged.is_symlink() or not staged.is_file() or not filecmp.cmp(
                            source_root / relative, staged, shallow=False):
                        raise ValueError(f"Worker source snapshot differs from original: {relative}")
            if any(path.is_symlink() or path.relative_to(source).as_posix() not in listed
                   for path in source.rglob("*") if path.is_file() or path.is_symlink()):
                raise ValueError("Worker source snapshot contains an unlisted file")
            if runtime["backend"] == "local":
                for staged in source.rglob("*"):
                    if staged.is_file():
                        staged.chmod(0o444)
        for name, path in task.inputs.items():
            destination = (workspace / "inputs" / name).resolve()
            if not destination.is_relative_to(workspace / "inputs"):
                raise ValueError("Worker input must stay under inputs/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        conversation_id = task.conversation_id or str(uuid.uuid5(uuid.NAMESPACE_URL,
                                                                  f"{task.task_id}:attempt:{task.attempt}"))
        request = task.model_dump(exclude={"source_manifest", "inputs"})
        # The frozen model identity is already OpenAI-compatible.  Do not add a
        # provider prefix here: local gateways expose the exact id returned by
        # /models and reject otherwise equivalent aliases.
        if runtime["backend"] == "docker":
            request.update(workspace="/workspace", source_root="/workspace/source",
                           conversation_id=conversation_id, base_url=_container_url(task.base_url),
                           construction_events_path="/workspace/logs/coding_events.jsonl")
        else:
            request.update(workspace=str(workspace), source_root=str(source),
                           conversation_id=conversation_id, base_url=task.base_url,
                           construction_events_path=str(workspace / "logs" / "coding_events.jsonl"))
        request["backend"] = runtime["backend"]
        request_path = workspace / "state" / "coding_task.json"
        result_path = workspace / "state" / f"coding_result_{task.kind}.json"
        if request_path.is_file() and json.loads(request_path.read_text(encoding="utf-8")) == request:
            if result_path.is_file():
                cached = CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
                if cached.status == "complete":
                    return self._validate_outputs(task, cached, workspace, conversation_id, runtime["backend"])
        result_path.unlink(missing_ok=True)
        atomic_write_json(request_path, request)

        key = get_llm_api_key()
        if runtime["backend"] == "local":
            forward_stop = threading.Event()
            forwarder = self._start_event_forwarder(workspace, task.construction_events_path, forward_stop)
            try:
                return self._run_local(task, workspace, request_path, result_path, conversation_id, key)
            finally:
                forward_stop.set()
                forwarder.join(timeout=2)
                self._forward_events(workspace, task.construction_events_path)
        image = settings["coding_worker_image"]
        try:
            image_id = docker_image_id(image)
        except ExecutionUnavailable as exc:
            if self.runtime is None:
                image_id = image
            else:
                return self._record_result(workspace, task, CodingResult(
                    status="blocked", conversation_id=conversation_id,
                    diagnostic=str(exc), error_category="environment_unavailable"))
        probe = (
            "import json,os,sys,urllib.request;"
            "url,model=sys.argv[1:3];"
            "headers={'Authorization':'Bearer '+os.environ['MEDBENCH_LLM_API_KEY']};"
            "body=json.load(urllib.request.urlopen(urllib.request.Request(url.rstrip('/')+'/models',"
            "headers=headers),timeout=15));"
            "assert model in [item['id'] for item in body['data']]"
        )
        environment = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1",
                       "MEDBENCH_LLM_API_KEY": key}
        try:
            check = subprocess.run(
                ["docker", "run", "--rm", "--add-host", "host.docker.internal:host-gateway",
                 "-e", "MEDBENCH_LLM_API_KEY", image_id, "python", "-c", probe,
                 request["base_url"], task.model],
                env=environment, capture_output=True, text=True, timeout=30, check=False)
            if check.returncode:
                return self._record_result(workspace, task, CodingResult(
                    status="blocked", conversation_id=conversation_id,
                    diagnostic="OpenHands container cannot reach the frozen Chat model",
                    error_category="model_unavailable"))
        except (OSError, subprocess.TimeoutExpired):
            return self._record_result(workspace, task, CodingResult(
                status="blocked", conversation_id=conversation_id,
                diagnostic="OpenHands container cannot probe the frozen Chat model",
                error_category="environment_unavailable"))
        forward_stop = threading.Event()
        forwarder = self._start_event_forwarder(workspace, task.construction_events_path, forward_stop)
        container_name = "medbench-worker-" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{task.task_id}:attempt:{task.attempt}").hex
        command = ["docker", "run", "-d", "--pull=never", "--name", container_name,
                   "--add-host", "host.docker.internal:host-gateway",
                   "-e", "MEDBENCH_LLM_API_KEY", "-w", "/workspace", "--mount",
                   _mount(source, "/workspace/source", readonly=True)]
        for name in ("inputs", "work", "outputs", "state", "logs", "conversations"):
            command.extend(["--mount", _mount(workspace / name, f"/workspace/{name}",
                                              readonly=name == "inputs")])
        command.extend([image_id, "python", "/app/openhands_worker.py",
                        "/workspace/state/coding_task.json",
                        f"/workspace/state/coding_result_{task.kind}.json"])
        try:
            existing = subprocess.run(["docker", "container", "inspect", container_name],
                                      capture_output=True, text=True, timeout=15, check=False)
            if existing.returncode:
                started = subprocess.run(command, env=environment, capture_output=True,
                                         text=True, timeout=30, check=False)
                if started.returncode:
                    return self._record_result(workspace, task, CodingResult(
                        status="failed", conversation_id=conversation_id,
                        diagnostic="OpenHands container could not start: " +
                        (started.stderr.strip()[:300] or "see Docker daemon"),
                        error_category="environment_unavailable", retryable=True))
            completed = subprocess.run(["docker", "wait", container_name], capture_output=True,
                                       text=True, timeout=task.timeout_seconds, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._record_result(workspace, task, CodingResult(
                status="failed", conversation_id=conversation_id,
                diagnostic=f"OpenHands worker unavailable: {type(exc).__name__}",
                error_category="worker_execution_failed", retryable=True))
        finally:
            try:
                with (workspace / "logs" / "stdout.log").open("a", encoding="utf-8") as stdout, \
                        (workspace / "logs" / "stderr.log").open("a", encoding="utf-8") as stderr:
                    subprocess.run(["docker", "logs", container_name], stdout=stdout, stderr=stderr,
                                   timeout=30, check=False)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True,
                               timeout=30, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        forward_stop.set()
        forwarder.join(timeout=2)
        self._forward_events(workspace, task.construction_events_path)
        if completed.returncode or (getattr(completed, "stdout", "0") or "0").strip() != "0" or not result_path.is_file():
            return self._record_result(workspace, task, CodingResult(
                status="failed", conversation_id=conversation_id,
                diagnostic=self._failure_summary(workspace / "logs" / "stderr.log"),
                error_category="worker_execution_failed", retryable=True))
        result = CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        return self._record_result(workspace, task,
                                   self._validate_outputs(task, result, workspace, conversation_id, runtime["backend"]))

    @staticmethod
    def _start_event_forwarder(workspace: Path, destination: str, stop: threading.Event):
        def loop():
            while not stop.is_set():
                OpenHandsCodingWorker._forward_events(workspace, destination)
                stop.wait(0.5)
        thread = threading.Thread(target=loop, name="openhands-event-forwarder", daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _record_result(workspace: Path, task: CodingTask, result: CodingResult) -> CodingResult:
        """Persist an immutable attempt receipt used by recovery and diagnosis."""
        receipt = {
            "task_id": task.task_id, "kind": task.kind, "attempt": task.attempt,
            "conversation_id": result.conversation_id, "status": result.status,
            "error_category": result.error_category, "retryable": result.retryable,
            "diagnostic": result.diagnostic, "outputs": result.outputs,
            "model": task.model, "base_url": task.base_url,
        }
        atomic_write_json(workspace / "state" / f"coding_attempt_{task.attempt}.json", receipt)
        return result

    def _run_local(self, task, workspace, request_path, result_path, conversation_id, key):
        runner = Path(__file__).resolve().parents[1] / "runtime" / "openhands_worker.py"
        command = ["uv", "run", "--no-project", "--python", "3.12"]
        for package in self._requirements:
            command.extend(["--with", package])
        command.extend(["python", str(runner), str(request_path), str(result_path)])
        environment = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1",
                       "MEDBENCH_LLM_API_KEY": key}
        try:
            timed_out = False
            with (workspace / "logs" / "stdout.log").open("a", encoding="utf-8") as stdout, \
                    (workspace / "logs" / "stderr.log").open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(
                    command, cwd=workspace, env=environment, stdout=stdout, stderr=stderr,
                    start_new_session=os.name != "nt",
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                )
                try:
                    process.wait(timeout=task.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, check=False, timeout=30)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if timed_out:
                return self._record_result(workspace, task, CodingResult(
                    status="failed", conversation_id=conversation_id,
                    diagnostic="OpenHands worker exceeded its time budget",
                    error_category="worker_execution_failed", retryable=True))
        except OSError as exc:
            return self._record_result(workspace, task, CodingResult(
                status="blocked", conversation_id=conversation_id,
                diagnostic=f"OpenHands local dependencies unavailable: {type(exc).__name__}",
                error_category="environment_unavailable"))
        self._forward_events(workspace, task.construction_events_path)
        if process.returncode or not result_path.is_file():
            return self._record_result(workspace, task, CodingResult(
                status="failed", conversation_id=conversation_id,
                diagnostic=self._failure_summary(workspace / "logs" / "stderr.log"),
                error_category="worker_execution_failed", retryable=True))
        result = CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        return self._record_result(workspace, task,
                                   self._validate_outputs(task, result, workspace, conversation_id, "local"))

    @staticmethod
    def _forward_events(workspace: Path, destination: str) -> None:
        events = workspace / "logs" / "coding_events.jsonl"
        marker = workspace / "state" / "events_forwarded.json"
        if not destination or not events.is_file():
            return
        offset = (json.loads(marker.read_text(encoding="utf-8"))["offset"]
                  if marker.is_file() else 0)
        with events.open("rb") as source_events, Path(destination).open("ab") as run_events:
            source_events.seek(offset)
            shutil.copyfileobj(source_events, run_events)
            atomic_write_json(marker, {"offset": source_events.tell()})

    @staticmethod
    def _failure_summary(stderr: Path) -> str:
        tail = ""
        if stderr.is_file():
            with stderr.open("rb") as handle:
                handle.seek(max(0, stderr.stat().st_size - 8000))
                tail = handle.read().decode("utf-8", errors="replace")
        if "ContextWindowExceeded" in tail or "LLMContextWindowExceedError" in tail:
            limit = re.search(r"maximum context length is (\d+)", tail)
            output = re.search(r"requested (\d+) output tokens", tail)
            input_tokens = re.search(r"prompt contains at least (\d+) input tokens", tail)
            if limit and output and input_tokens:
                return (f"CodingWorker context window exceeded: input={input_tokens[1]}, "
                        f"reserved_output={output[1]}, model_limit={limit[1]}; see logs/stderr.log")
            return "CodingWorker context window exceeded; see logs/stderr.log for details"
        return "OpenHands worker failed; see logs/stderr.log for details"

    @staticmethod
    def _validate_outputs(task: CodingTask, result: CodingResult, workspace: Path,
                          conversation_id: str, backend: str) -> CodingResult:
        if result.conversation_id != conversation_id:
            raise ValueError("Worker returned another conversation identity")
        if result.status == "complete" and set(result.outputs) != set(task.required_outputs):
            raise ValueError("Worker did not return every required output")
        outputs = {}
        for name, path in result.outputs.items():
            expected = workspace / "outputs" / name
            valid_path = (path == f"/workspace/outputs/{name}" if backend == "docker"
                          else Path(path).resolve() == expected.resolve())
            if name not in task.required_outputs or not valid_path or not expected.is_file():
                raise ValueError(f"Worker output is outside its contract: {name}")
            outputs[name] = str(expected.resolve())
        verify_source(task.source_manifest)
        return result.model_copy(update={"outputs": outputs})


def coding_worker(context) -> CodingWorker:
    return OpenHandsCodingWorker(context.get("execution_runtime"))
