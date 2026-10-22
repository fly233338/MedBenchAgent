"""Generic Coding Agent: Workstream-local SDK execution and Host-owned file handoffs."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid

from filelock import FileLock, Timeout
from langgraph.runtime import Runtime

from medbenchagent.runtime.construction import record_event
from medbenchagent.settings import CodingSettings, get_coding_api_key

from .artifacts import ArtifactIntegrityError, _within, resolve_artifact, write_artifact
from .coding_worker import CandidateNeed, CodingCandidate, failure
from .contracts import WorkNeed, WorkRequest, WorkResult
from .interfaces import AgentInput, AgentOutput, AgentSpec, CompilationContext


def _stop_process(process):
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=30, check=False,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=10)


def run_local_worker(payload: dict) -> CodingCandidate:
    request = WorkRequest.model_validate(payload["request"])
    runtime = payload["settings"]["coding"]
    python = Path(runtime["worker_python"])
    if not python.is_file():
        return failure(request.request_id, "blocked", "environment_unavailable",
                       f"Coding interpreter does not exist: {python}")
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                   "OPENHANDS_SUPPRESS_BANNER": "1", "MEDBENCH_CODING_LLM_API_KEY": get_coding_api_key()}
    environment.pop("MEDBENCH_LLM_API_KEY", None)
    environment["PATH"] = os.pathsep.join((str(python.parent), environment.get("PATH", "")))
    environment["VIRTUAL_ENV"] = str(python.parent.parent)
    # The separate environment needs only this package's lightweight worker and contracts.
    source = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source, environment.get("PYTHONPATH"))))
    log = Path(payload["log_path"])
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        record_event(log, "coding_request", compilation_id=request.compilation_id,
                     request_id=request.request_id, agent_id=request.agent_id,
                     workstream_id=request.workstream_id)
        with log.open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                [str(python), "-m", "medbenchagent.compilation.coding_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", cwd=payload["workspace"], env=environment,
                start_new_session=os.name != "nt",
                creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                               if os.name == "nt" else 0),
            )
            try:
                stdout, _ = process.communicate(json.dumps(payload, ensure_ascii=False),
                                                timeout=runtime["timeout_seconds"])
            except subprocess.TimeoutExpired:
                _stop_process(process)
                return failure(request.request_id, "failed", "coding_timeout",
                               "Coding worker exceeded its frozen time budget")
            except BaseException:
                _stop_process(process)
                raise
    except OSError as exc:
        return failure(request.request_id, "blocked", "environment_unavailable",
                       f"Cannot start Coding worker ({type(exc).__name__})")
    if process.returncode:
        return failure(request.request_id, "failed", "worker_exit",
                       f"Coding worker exited with code {process.returncode}; see request log")
    try:
        return CodingCandidate.model_validate_json(stdout)
    except ValueError:
        return failure(request.request_id, "failed", "invalid_submission",
                       "Coding worker did not return a valid candidate result; see request log")


class OpenHandsCodingAgent:
    def __init__(self, settings: CodingSettings, *, instructions: str, runner: Callable | None = None):
        if not settings.llm.model:
            raise ValueError("Resolve the Coding model before starting a run")
        self.settings = settings
        self.instructions = instructions
        self.runner = runner or run_local_worker

    def spec(self) -> AgentSpec:
        return AgentSpec(
            "OpenHands general-purpose Coding Agent. Explores supplied files, writes and executes "
            "code, checks results, and submits files or reports needs/failures. Select a Workstream "
            "to resume its private working context, or create a new one. Share work only via artifacts.",
            self, uses_workstreams=True,
        )

    def __call__(self, state: AgentInput, runtime: Runtime[CompilationContext]) -> AgentOutput:
        request = WorkRequest.model_validate(state["request"])
        directory = runtime.context.directory.resolve()
        try:
            result = self._execute(directory, request, runtime.context.goal)
        except ArtifactIntegrityError as exc:
            result = self._result(failure(request.request_id, "failed", "artifact_integrity", str(exc)), [])
        except (ValueError, OSError) as exc:
            result = WorkResult(
                **failure(request.request_id, "failed", "invalid_handoff", str(exc)).model_dump(
                    exclude={"files", "needs"}),
            )
        return {"result": result.model_dump(mode="json")}

    def _execute(self, directory: Path, request: WorkRequest, goal: str) -> WorkResult:
        if request.workstream_id is None:
            raise ValueError("Coding requests require a Host-assigned Workstream ID")
        root = _within(directory, f"coding/{request.workstream_id}")
        workspace = _within(root, "workspace")
        output = _within(workspace, f"outputs/{request.request_id}")
        inputs = []
        for ref in request.inputs:
            try:
                path = resolve_artifact(directory, ref)
            except FileNotFoundError:
                candidate = failure(request.request_id, "needs_input", "input_missing",
                                    f"Registered input is missing: {ref.artifact_id}")
                candidate.needs.append(CandidateNeed(
                    need_id=f"{request.request_id}-input", kind="work",
                    summary=f"Restore or replace registered input {ref.artifact_id}",
                ))
                return self._result(candidate, [])
            if not path.is_relative_to(_within(directory, "artifacts")):
                raise ValueError("Coding inputs must be formal artifacts, not workspace files")
            inputs.append({**ref.model_dump(mode="json"), "read_path": str(path)})
        conversation_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                        f"{request.compilation_id}/{request.agent_id}/{request.workstream_id}"))
        details = {
            "compilation_goal": goal,
            "request": request.model_dump(mode="json"), "inputs": inputs,
            "workspace": str(workspace), "output_directory": str(output),
        }
        payload = {
            **details, "conversation_id": conversation_id,
            "persistence_dir": str(_within(root, "conversations")),
            "log_path": str(_within(root, f"logs/{request.request_id}.log")),
            "settings": self.settings.snapshot(),
            "prompt": self.instructions + "\n" + json.dumps(details, ensure_ascii=False, sort_keys=True),
        }
        root.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(root / ".lock"), timeout=0):
                output.mkdir(parents=True, exist_ok=True)
                candidate = CodingCandidate.model_validate(self.runner(payload))
                if candidate.request_id != request.request_id:
                    raise ValueError("Coding candidate belongs to another request")
                # Validate every candidate before publishing any file.
                files, seen = [], set()
                for item in candidate.files:
                    actual = _within(output, item.path)
                    if not actual.is_file() or actual in seen:
                        raise ValueError(f"Missing or duplicate Coding output: {item.path}")
                    seen.add(actual)
                    files.append((actual.relative_to(output).as_posix(), item.kind, actual.read_bytes()))
                artifacts = [write_artifact(directory, request, name, kind=kind, content=content)
                             for name, kind, content in files]
                return self._result(candidate, artifacts)
        except Timeout:
            return self._result(failure(request.request_id, "blocked", "workstream_busy",
                                        "This Workstream is already executing another request"), [])

    @staticmethod
    def _result(candidate, artifacts):
        return WorkResult(
            request_id=candidate.request_id, outcome=candidate.outcome, summary=candidate.summary,
            artifacts=artifacts, issues=candidate.issues,
            needs=[WorkNeed(**need.model_dump()) for need in candidate.needs],
        )
