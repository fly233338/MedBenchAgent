"""OpenHands preparation work with a read-only source mount."""

from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from medbenchagent.preparation.contracts import StageRequest, StageResult
from medbenchagent.data.execution import ExecutionUnavailable, docker_image_id
from medbenchagent.data.sources import digest_file
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_llm_api_key, get_settings


WORKER_STARTUP_TIMEOUT_SECONDS = 180


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


def _worker_runtime_error(request: StageRequest) -> str:
    runtime = request.worker_runtime or {}
    executable = Path(str(runtime.get("python") or ""))
    if not executable.is_file():
        return f"Coding worker interpreter is missing: {executable}"
    requirements_digest = str(runtime.get("requirements_digest") or "")
    requirements = Path(__file__).resolve().parents[3] / "config" / "coding-worker.requirements.txt"
    if requirements_digest and requirements.is_file() and digest_file(requirements) != requirements_digest:
        return "Coding worker requirements changed after the run was frozen"
    try:
        probe = subprocess.run(
            [str(executable), "-c", "import openhands.sdk"],
            capture_output=True, text=True, timeout=WORKER_STARTUP_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Coding worker interpreter cannot start: {type(exc).__name__}"
    if probe.returncode:
        return "Coding worker interpreter is missing OpenHands dependencies"
    return ""


class OpenHandsDatasetRuntime:
    """Run one OpenHands coding stage on the backend frozen for the parent run."""

    _active: dict[str, tuple[str, object]] = {}
    _event_forward_lock = threading.Lock()

    @classmethod
    def cancel_active_workers(cls) -> None:
        for conversation_id in list(cls._active):
            cls().cancel(conversation_id)

    def __init__(self, runtime=None):
        self.runtime = runtime

    def _failure(self, request: StageRequest, status: str, message: str, reason: str) -> StageResult:
        return StageResult(stage=request.stage, run_id=request.run_id, dataset_id=request.dataset_id, attempt=request.attempt,
                           attempt_id=request.attempt_id, conversation_id=request.conversation_id,
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
        if runtime.get("backend") == "local":
            if diagnostic := _worker_runtime_error(request):
                return self._record_result(
                    workspace, request,
                    self._failure(request, "blocked", diagnostic, "environment_unavailable"),
                )
        source_roots = {str(key): str(value) for key, value in request.source_roots.items()}
        if not source_roots:
            raise ValueError("Stage request requires at least one source root")
        primary_id = sorted(source_roots)[0]
        source_root = Path(source_roots[primary_id]).resolve()
        if any(workspace.is_relative_to(Path(root).resolve()) or Path(root).resolve().is_relative_to(workspace)
               for root in source_roots.values()):
            raise ValueError("Worker workspace and original source must be separate")
        for name in ("work/inputs", "work/outputs", "work/logs", "work/conversations"):
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
            agent=request.agent,
            backend=runtime["backend"],
            previous_attempt_root=request.previous_attempt_root,
            worker_runtime=request.worker_runtime,
        )
        if runtime["backend"] == "docker":
            deployment.update(
                base_url=_container_url(request.base_url),
                construction_events_path="/workspace/work/logs/coding_events.jsonl",
                context_path="/workspace/work/inputs/context.json",
                prompt_path="/workspace/prompt.md",
            )
        else:
            deployment.update(
                base_url=request.base_url,
                construction_events_path=str(workspace / "work" / "logs" / "coding_events.jsonl"),
                context_path=str((workspace / "work" / "inputs" / "context.json").resolve()),
                prompt_path=str((workspace / "prompt.md").resolve()),
            )
        request_path = workspace / "work" / "deployment.json"
        result_path = workspace / "result.json"
        if request_path.is_file() and json.loads(request_path.read_text(encoding="utf-8")) == deployment:
            if result_path.is_file():
                cached = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
                if cached.execution_key == request.execution_key:
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
        environment = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MEDBENCH_LLM_API_KEY": key}
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
        if request.previous_attempt_root:
            command.extend(["--mount", _mount(Path(request.previous_attempt_root).resolve(), "/workspace/previous", readonly=True)])
        command.extend(["--mount", _mount(workspace, "/workspace"),
                        "--mount", _mount(workspace / "work" / "inputs", "/workspace/work/inputs", readonly=True)])
        command.extend([image_id, "python", "/app/openhands_worker.py", "/workspace/work/deployment.json", "/workspace/result.json"])
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
            subprocess.run(["docker", "wait", container_name], capture_output=True,
                           text=True, timeout=request.budget.timeout_seconds, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._record_result(workspace, request, self._failure(
                request, "failed", f"OpenHands worker unavailable: {type(exc).__name__}", "worker_execution_failed"))
        finally:
            self._active.pop(conversation_id, None)
            try:
                with (workspace / "work" / "logs" / "stdout.log").open("a", encoding="utf-8") as stdout, \
                        (workspace / "work" / "logs" / "stderr.log").open("a", encoding="utf-8") as stderr:
                    subprocess.run(["docker", "logs", container_name], stdout=stdout, stderr=stderr, timeout=30, check=False)
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        forward_stop.set()
        forwarder.join(timeout=2)
        self._forward_events(workspace, request.construction_events_path)
        return self._result_after_process(
            request, workspace, result_path, conversation_id, runtime["backend"],
        )

    @staticmethod
    def _start_event_forwarder(workspace: Path, destination: str, stop: threading.Event):
        def loop():
            while not stop.is_set():
                OpenHandsDatasetRuntime._forward_events(workspace, destination)
                stop.wait(0.5)
        thread = threading.Thread(target=loop, name="openhands-event-forwarder", daemon=True)
        thread.start()
        return thread

    def _result_after_process(self, request: StageRequest, workspace: Path, result_path: Path,
                              conversation_id: str, backend: str) -> StageResult:
        """Preserve a worker's structured failure before falling back to stderr."""
        if result_path.is_file():
            try:
                result = StageResult.model_validate_json(result_path.read_text(encoding="utf-8"))
                if result.conversation_id != conversation_id:
                    raise ValueError("Worker returned another conversation identity")
                if result.status != "submitted":
                    return result
                return self._record_result(workspace, request, self._validate_outputs(request, result, workspace, conversation_id, backend))
            except (OSError, ValueError, TypeError):
                pass
        return self._record_result(workspace, request, self._failure(
            request, "failed", self._failure_summary(workspace / "work" / "logs" / "stderr.log"), "worker_execution_failed"))

    @staticmethod
    def _record_result(workspace: Path, request: StageRequest, result: StageResult) -> StageResult:
        result = result.model_copy(update={"attempt_id": request.attempt_id, "execution_key": request.execution_key})
        atomic_write_json(workspace / "result.json", result.model_dump(mode="json"))
        return result

    @staticmethod
    def _local_command(request, runner: Path, request_path: Path, result_path: Path) -> list[str]:
        worker_python = str((request.worker_runtime or {}).get("python") or "")
        return [worker_python, str(runner), str(request_path), str(result_path)]

    def _run_local(self, request, workspace, request_path, result_path, conversation_id, key):
        runner = Path(__file__).resolve().parents[1] / "runtime" / "openhands_worker.py"
        command = self._local_command(request, runner, request_path, result_path)
        environment = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1",
                       "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                       "MEDBENCH_LLM_API_KEY": key}
        try:
            timed_out = False
            with (workspace / "work" / "logs" / "stdout.log").open("a", encoding="utf-8") as stdout, \
                    (workspace / "work" / "logs" / "stderr.log").open("a", encoding="utf-8") as stderr:
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
        return self._result_after_process(request, workspace, result_path, conversation_id, "local")

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
        with OpenHandsDatasetRuntime._event_forward_lock:
            OpenHandsDatasetRuntime._forward_events_unlocked(workspace, destination)

    @staticmethod
    def _forward_events_unlocked(workspace: Path, destination: str) -> None:
        events = workspace / "work" / "logs" / "coding_events.jsonl"
        marker = workspace / "work" / "events_forwarded.json"
        if not destination or not events.is_file():
            return
        saved = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        offset = int(saved.get("offset", 0))
        size = events.stat().st_size
        if size < offset:
            offset = 0
            saved["source_generation"] = int(saved.get("source_generation", 0)) + 1
        with events.open("rb") as source_events:
            source_events.seek(offset)
            data = source_events.read()
            complete_length = data.rfind(b"\n") + 1
            complete = data[:complete_length]
            new_offset = offset + complete_length
        if not complete:
            atomic_write_json(marker, {**saved, "offset": offset})
            return
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("ab") as run_events:
            for raw_line in complete.splitlines(keepends=True):
                line = raw_line.rstrip(b"\r\n")
                if not line:
                    continue
                try:
                    payload = json.loads(line.decode("utf-8", errors="strict"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = {"event_type": "invalid_event", "decode_error": True, "raw_digest": hashlib.sha256(line).hexdigest()}
                if isinstance(payload, dict):
                    payload.setdefault("event_id", hashlib.sha256(line).hexdigest())
                    payload.setdefault("source_generation", saved.get("source_generation", 0))
                    run_events.write((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        atomic_write_json(marker, {**saved, "offset": new_offset})

    @staticmethod
    def _failure_summary(stderr: Path) -> str:
        tail = ""
        if stderr.is_file():
            with stderr.open("rb") as handle:
                handle.seek(max(0, stderr.stat().st_size - 8000))
                raw = handle.read()
                try:
                    tail = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    return "OpenHands worker emitted invalid UTF-8 in stderr; see logs/stderr.log for raw bytes"
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
            expected = workspace / "work" / "outputs" / name
            valid_path = (path == f"/workspace/work/outputs/{name}" if backend == "docker"
                          else Path(path).resolve() == expected.resolve())
            relative_name = Path(name)
            if relative_name.is_absolute() or ".." in relative_name.parts or not valid_path or not expected.is_file():
                raise ValueError(f"Worker output is outside its contract: {name}")
            outputs[name] = str(expected.resolve())
        return result.model_copy(update={"outputs": outputs})


def dataset_runtime(context) -> DatasetRuntimeProtocol:
    return OpenHandsDatasetRuntime(context.get("execution_runtime"))


