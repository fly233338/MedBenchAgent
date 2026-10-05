"""OpenHands preparation work with a read-only source mount."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import signal
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from medbenchagent.preparation.contracts import StageRequest, StageResult

from medbenchagent.data.sources import verify_source
from medbenchagent.data.execution import ExecutionUnavailable, docker_image_id
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_llm_api_key, get_settings


class DatasetRuntimeProtocol(Protocol):
    def run(self, request: StageRequest) -> StageResult: ...


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


def _worker_llm_error(request: StageRequest) -> str:
    """Return a deterministic configuration error before starting OpenHands."""
    if not request.model.strip():
        return "Coding worker model is missing from the run-frozen stage request"
    parsed = urlsplit(request.base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Coding worker base_url is missing or is not an HTTP(S) URL"
    return ""


class OpenHandsDatasetRuntime:
    """Run one OpenHands coding stage on the backend frozen for the parent run."""

    _requirements = (
        "openhands-sdk==1.49.6", "openhands-tools==1.49.6", "pandas==2.3.3",
        "openpyxl==3.1.5", "xlrd==2.0.2", "duckdb==1.5.4", "Pillow==12.3.0",
        "pydicom==3.0.1", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg==2.3.0",
        "pylibjpeg-openjpeg==2.5.0", "nibabel==5.3.2", "pynrrd==1.1.3",
        "SimpleITK==2.5.3",
    )
    _active: dict[str, tuple[str, object]] = {}

    @classmethod
    def cancel_active_workers(cls) -> None:
        for conversation_id in list(cls._active):
            cls().cancel(conversation_id)

    def __init__(self, runtime=None):
        self.runtime = runtime

    def _failure(self, request: StageRequest, status: str, message: str, reason: str) -> StageResult:
        return StageResult(stage=request.stage, attempt=request.attempt, conversation_id=request.conversation_id,
                           execution_key=request.execution_key, status=status, diagnostic=message,
                           termination_reason=reason, input_digest=request.input_digest,
                           prompt_digest=hashlib.sha256(Path(request.prompt_path).read_bytes()).hexdigest()
                           if Path(request.prompt_path).is_file() else "",
                           context_digest=hashlib.sha256(Path(request.context_path).read_bytes()).hexdigest()
                           if Path(request.context_path).is_file() else "",
                           token_count=request.prompt_token_count)

    def run(self, request: StageRequest) -> StageResult:
        settings = get_settings().data_processing
        runtime = self.runtime or {"backend": "docker", "requested_backend": "docker"}
        workspace = Path(request.workspace).resolve()
        if diagnostic := _worker_llm_error(request):
            return self._record_result(
                workspace, request,
                self._failure(request, "blocked", diagnostic, "model_unavailable"),
            )
        source_roots = {str(key): str(value) for key, value in request.source_roots.items()}
        if not source_roots:
            raise ValueError("Stage request requires at least one source root")
        for manifest in request.source_manifests.values():
            verify_source(manifest)
        primary_id = sorted(source_roots)[0]
        source_root = Path(source_roots[primary_id]).resolve()
        if any(workspace.is_relative_to(Path(root).resolve()) or Path(root).resolve().is_relative_to(workspace)
               for root in source_roots.values()):
            raise ValueError("Worker workspace and original source must be separate")
        for name in ("inputs", "work", "outputs", "state", "logs", "conversations"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        self._forward_events(workspace, request.construction_events_path)
        conversation_id = request.conversation_id
        stage_value = request.stage.value
        deployment = request.model_dump(mode="json")
        deployment.update(
            task_id=f"{request.run_id}:{request.dataset_id}:{stage_value}",
            kind="preparation",
            source_id=primary_id,
            stage=stage_value,
            workflow_step=stage_value,
            source_root="/workspace/source" if runtime["backend"] == "docker" else str(source_root),
            workspace="/workspace" if runtime["backend"] == "docker" else str(workspace),
            conversation_id=conversation_id,
            source_digest=request.input_digest,
            max_iterations=request.budget.max_iterations,
            timeout_seconds=request.budget.timeout_seconds,
            allowed_outputs=request.required_outputs + request.optional_outputs,
            progress_guard={"no_progress_limit": request.budget.no_progress_limit,
                            "tool_error_limit": request.budget.tool_error_limit},
            agent=request.agent,
            backend=runtime["backend"],
        )
        if runtime["backend"] == "docker":
            deployment.update(
                base_url=_container_url(request.base_url),
                construction_events_path="/workspace/logs/coding_events.jsonl",
                context_path="/workspace/inputs/context.json",
                prompt_path="/workspace/inputs/prompt.md",
            )
        else:
            deployment.update(
                base_url=request.base_url,
                construction_events_path=str(workspace / "logs" / "coding_events.jsonl"),
                context_path=str((workspace / "inputs" / "context.json").resolve()),
                prompt_path=str((workspace / "inputs" / "prompt.md").resolve()),
            )
        request_path = workspace / "state" / "request.json"
        result_path = workspace / "state" / "result.json"
        if request.execution_key:
            receipt_path = workspace / "state" / f"coding_attempt_{request.attempt}.json"
            if receipt_path.is_file():
                try:
                    cached = StageResult.model_validate_json(receipt_path.read_text(encoding="utf-8"))
                    if cached.execution_key == request.execution_key and cached.status == "submitted":
                        return self._validate_outputs(request, cached, workspace, conversation_id, runtime["backend"])
                except ValueError:
                    pass
        if request_path.is_file() and json.loads(request_path.read_text(encoding="utf-8")) == deployment:
            if result_path.is_file():
                cached = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
                if cached.status == "submitted":
                    return self._validate_outputs(request, cached, workspace, conversation_id, runtime["backend"])
        result_path.unlink(missing_ok=True)
        atomic_write_json(request_path, deployment)

        key = get_llm_api_key()
        if runtime["backend"] == "local":
            forward_stop = threading.Event()
            forwarder = self._start_event_forwarder(workspace, request.construction_events_path, forward_stop)
            try:
                return self._run_local(request, workspace, request_path, result_path, conversation_id, key)
            finally:
                self._active.pop(conversation_id, None)
                forward_stop.set()
                forwarder.join(timeout=2)
                self._forward_events(workspace, request.construction_events_path)

        image = settings["coding_worker_image"]
        try:
            image_id = docker_image_id(image)
        except ExecutionUnavailable as exc:
            if self.runtime is None:
                image_id = image
            else:
                return self._record_result(workspace, request, self._failure(request, "blocked", str(exc), "environment_unavailable"))
        probe = (
            "import json,os,sys,urllib.request;"
            "url,model=sys.argv[1:3];"
            "headers={'Authorization':'Bearer '+os.environ['MEDBENCH_LLM_API_KEY']};"
            "body=json.load(urllib.request.urlopen(urllib.request.Request(url.rstrip('/')+'/models',"
            "headers=headers),timeout=15));"
            "assert model in [item['id'] for item in body['data']]"
        )
        environment = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1", "MEDBENCH_LLM_API_KEY": key}
        try:
            check = subprocess.run(
                ["docker", "run", "--rm", "--add-host", "host.docker.internal:host-gateway",
                 "-e", "MEDBENCH_LLM_API_KEY", image_id, "python", "-c", probe,
                 deployment["base_url"], request.model],
                env=environment, capture_output=True, text=True, timeout=30, check=False)
            if check.returncode:
                return self._record_result(workspace, request, self._failure(
                    request, "blocked", "OpenHands container cannot reach the frozen Chat model", "model_unavailable"))
        except (OSError, subprocess.TimeoutExpired):
            return self._record_result(workspace, request, self._failure(
                request, "blocked", "OpenHands container cannot probe the frozen Chat model", "environment_unavailable"))
        forward_stop = threading.Event()
        forwarder = self._start_event_forwarder(workspace, request.construction_events_path, forward_stop)
        container_name = "medbench-worker-" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{deployment['task_id']}:attempt:{request.attempt}").hex
        command = ["docker", "run", "-d", "--pull=never", "--name", container_name,
                   "--add-host", "host.docker.internal:host-gateway", "-e", "MEDBENCH_LLM_API_KEY",
                   "-w", "/workspace", "--mount", _mount(source_root, "/workspace/source", readonly=True)]
        for source_id, root in source_roots.items():
            command.extend(["--mount", _mount(Path(root).resolve(), f"/workspace/sources/{source_id}", readonly=True)])
        for name in ("inputs", "work", "outputs", "state", "logs", "conversations"):
            command.extend(["--mount", _mount(workspace / name, f"/workspace/{name}", readonly=name == "inputs")])
        command.extend([image_id, "python", "/app/openhands_worker.py", "/workspace/state/request.json", "/workspace/state/result.json"])
        try:
            existing = subprocess.run(["docker", "container", "inspect", container_name],
                                      capture_output=True, text=True, timeout=15, check=False)
            if existing.returncode:
                started = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30, check=False)
                if started.returncode:
                    return self._record_result(workspace, request, self._failure(
                        request, "failed", "OpenHands container could not start: " +
                        (started.stderr.strip()[:300] or "see Docker daemon"), "environment_unavailable"))
            self._active[conversation_id] = ("docker", container_name)
            completed = subprocess.run(["docker", "wait", container_name], capture_output=True,
                                       text=True, timeout=request.budget.timeout_seconds, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._record_result(workspace, request, self._failure(
                request, "failed", f"OpenHands worker unavailable: {type(exc).__name__}", "worker_execution_failed"))
        finally:
            self._active.pop(conversation_id, None)
            try:
                with (workspace / "logs" / "stdout.log").open("a", encoding="utf-8") as stdout, \
                        (workspace / "logs" / "stderr.log").open("a", encoding="utf-8") as stderr:
                    subprocess.run(["docker", "logs", container_name], stdout=stdout, stderr=stderr, timeout=30, check=False)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        forward_stop.set()
        forwarder.join(timeout=2)
        self._forward_events(workspace, request.construction_events_path)
        if completed.returncode or (getattr(completed, "stdout", "0") or "0").strip() != "0" or not result_path.is_file():
            return self._record_result(workspace, request, self._failure(
                request, "failed", self._failure_summary(workspace / "logs" / "stderr.log"), "worker_execution_failed"))
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        return self._record_result(request_workspace := workspace, request,
                                   self._validate_outputs(request, result, request_workspace, conversation_id, runtime["backend"]))

    @staticmethod
    def _start_event_forwarder(workspace: Path, destination: str, stop: threading.Event):
        def loop():
            while not stop.is_set():
                OpenHandsDatasetRuntime._forward_events(workspace, destination)
                stop.wait(0.5)
        thread = threading.Thread(target=loop, name="openhands-event-forwarder", daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _record_result(workspace: Path, request: StageRequest, result: StageResult) -> StageResult:
        """Persist an immutable attempt receipt used by recovery and diagnosis."""
        def file_digest(path: str) -> str:
            candidate = Path(path)
            if not candidate.is_file():
                return ""
            import hashlib
            return hashlib.sha256(candidate.read_bytes()).hexdigest()

        receipt = {
            "stage": request.stage.value, "attempt": request.attempt,
            "conversation_id": result.conversation_id, "status": result.status,
            "diagnostic": result.diagnostic, "outputs": result.outputs,
            "model": request.model, "base_url": request.base_url,
            "execution_key": request.execution_key,
            "input_digest": request.input_digest,
            "prompt_token_count": request.prompt_token_count,
            "prompt_digest": file_digest(request.prompt_path),
            "context_digest": file_digest(request.context_path),
        }
        atomic_write_json(workspace / "state" / f"coding_attempt_{request.attempt}.json", receipt)
        return result

    def _run_local(self, request, workspace, request_path, result_path, conversation_id, key):
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
                self._active[conversation_id] = ("local", process)
                try:
                    process.wait(timeout=request.budget.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, check=False, timeout=30)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if timed_out:
                return self._record_result(workspace, request, self._failure(
                    request, "failed", "OpenHands worker exceeded its time budget", "worker_execution_failed"))
        except OSError as exc:
            return self._record_result(workspace, request, self._failure(
                request, "blocked", f"OpenHands local dependencies unavailable: {type(exc).__name__}", "environment_unavailable"))
        self._forward_events(workspace, request.construction_events_path)
        if process.returncode or not result_path.is_file():
            return self._record_result(workspace, request, self._failure(
                request, "failed", self._failure_summary(workspace / "logs" / "stderr.log"), "worker_execution_failed"))
        result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        return self._record_result(workspace, request,
                                   self._validate_outputs(request, result, workspace, conversation_id, "local"))

    def cancel(self, conversation_id: str) -> None:
        active = self._active.get(conversation_id)
        if not active:
            return
        backend, handle = active
        try:
            if backend == "docker":
                subprocess.run(["docker", "rm", "-f", str(handle)], capture_output=True, timeout=30, check=False)
            elif os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(handle.pid), "/T", "/F"], capture_output=True, timeout=30, check=False)
            else:
                os.killpg(handle.pid, signal.SIGKILL)
        finally:
            self._active.pop(conversation_id, None)

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
                return (f"DatasetRuntimeProtocol context window exceeded: input={input_tokens[1]}, "
                        f"reserved_output={output[1]}, model_limit={limit[1]}; see logs/stderr.log")
            return "DatasetRuntimeProtocol context window exceeded; see logs/stderr.log for details"
        return "OpenHands worker failed; see logs/stderr.log for details"

    @staticmethod
    def _validate_outputs(request: StageRequest, result: StageResult, workspace: Path,
                          conversation_id: str, backend: str) -> StageResult:
        if result.conversation_id != conversation_id:
            raise ValueError("Worker returned another conversation identity")
        if result.status == "submitted" and not set(request.required_outputs) <= set(result.outputs):
            raise ValueError("Worker did not return every required output")
        outputs = {}
        for name, path in result.outputs.items():
            expected = workspace / "outputs" / name
            valid_path = (path == f"/workspace/outputs/{name}" if backend == "docker"
                          else Path(path).resolve() == expected.resolve())
            relative_name = Path(name)
            if relative_name.is_absolute() or ".." in relative_name.parts or not valid_path or not expected.is_file():
                raise ValueError(f"Worker output is outside its contract: {name}")
            outputs[name] = str(expected.resolve())
        for manifest in request.source_manifests.values():
            verify_source(manifest)
        return result.model_copy(update={"outputs": outputs})


def dataset_runtime(context) -> DatasetRuntimeProtocol:
    return OpenHandsDatasetRuntime(context.get("execution_runtime"))


