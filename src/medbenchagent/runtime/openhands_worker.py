"""OpenHands SDK process entrypoint for one resumable coding task."""

from __future__ import annotations

import json
import filecmp
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def normalize_tool_arguments(tool_name: str, arguments: dict) -> tuple[dict, list[str]]:
    """Strip harmless gateway metadata before a tool retry."""
    normalized = dict(arguments or {})
    stripped = []
    if tool_name in {"terminal", "file_editor"} and "description_note" in normalized:
        normalized.pop("description_note", None)
        stripped.append("description_note")
    return normalized, stripped


def validate_terminal_command(command: str) -> str:
    """Reject dependency installation and unrelated network access in workers."""
    lowered = (command or "").casefold()
    forbidden = ("pip install", "python -m pip", "uv pip", "curl ", "wget ", "invoke-webrequest")
    if any(token in lowered for token in forbidden):
        return "Package installation and unrelated network access are disabled in preparation workers"
    return ""


def _litellm_model(model: str) -> str:
    """Return the provider-qualified model name required by LiteLLM."""
    return model if "/" in model else "openai/" + model


def _write_session_state(workspace: Path, request: dict, *, status: str, summary: str = "",
                         termination_reason: str = "", progress: dict | None = None) -> None:
    bounded = (summary or "")[:6000]
    state = {"schema": "worker-attempt", "task_id": request["task_id"],
             "kind": request["kind"], "source_root": request["source_root"],
             "source_digest": request.get("source_digest", ""),
             "workflow_step": request.get("workflow_step", ""),
             "stage": request.get("stage", request.get("workflow_step", "prepare")),
             "conversation_id": request["conversation_id"], "attempt": request.get("attempt", 1),
             "status": status, "last_feedback": {"message": bounded} if bounded else {},
             "stop_reason": bounded if status in {"blocked", "failed"} else "",
             "required_outputs": request.get("required_outputs", []),
             "termination_reason": termination_reason,
             "progress": progress or {}}
    state_dir = workspace / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "worker.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def replay_source_script(request: dict, workspace: Path) -> str:
    """Re-run the worker's inspection code twice before accepting its files."""
    script = workspace / "outputs" / "inspect_source.py"
    if not script.is_file():
        return "Required inspection script is missing"
    replay_marker = workspace / "state" / "source_replayed.json"
    replay_marker.unlink(missing_ok=True)
    replay = []
    source_root = Path(request.get("source_root", workspace / "source")).resolve()
    with tempfile.TemporaryDirectory(dir=workspace / "work") as first, \
            tempfile.TemporaryDirectory(dir=workspace / "work") as second:
        for trial in (Path(first), Path(second)):
            environment = {**os.environ, "MEDBENCH_SOURCE_ROOT": str(source_root),
                           "MEDBENCH_WORK_DIR": str(trial)}
            try:
                completed = subprocess.run([sys.executable, str(script)], cwd=workspace,
                                           env=environment, capture_output=True, text=True,
                                           encoding="utf-8", timeout=300, check=False)
            except subprocess.TimeoutExpired:
                return "Inspection script exceeded its 300-second replay budget"
            if completed.returncode:
                return "Inspection script replay failed: " + completed.stderr[-1500:]
            artifacts = [trial / "source_report.json", trial / "image_population.jsonl"]
            if any(not path.is_file() for path in artifacts):
                return "Inspection script must write both outputs under MEDBENCH_WORK_DIR"
            replay.append(artifacts)
        if any(not filecmp.cmp(left, right, shallow=False)
               for left, right in zip(replay[0], replay[1], strict=True)):
            return "Inspection script produced different outputs on replay"
        for name in ("source_report.json", "image_population.jsonl"):
            shutil.copyfile(Path(first) / name, workspace / "outputs" / name)
    replay_marker.write_text(json.dumps({"task_id": request["task_id"]}), encoding="utf-8")
    return ""


def main(request_path: str, result_path: str) -> None:
    from pydantic import SecretStr
    from openhands.sdk import Agent, Conversation, LLM, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    workspace = Path(request["workspace"]).resolve()
    _write_session_state(workspace, request, status="running")
    key = os.getenv("MEDBENCH_LLM_API_KEY")
    if not key:
        raise RuntimeError("Coding worker requires the configured LLM API key")
    # Keep the credential in the LLM object, away from terminal/file-editor subprocesses.
    os.environ.pop("MEDBENCH_LLM_API_KEY", None)
    # The run freezes the exact gateway model id.  LiteLLM needs an explicit
    # provider at this adapter boundary; the gateway still receives the raw id.
    model_name = str(request.get("model") or "").strip()
    base_url = str(request.get("base_url") or "").strip()
    if not model_name or not base_url:
        raise RuntimeError("Coding worker requires a non-empty run-frozen model and base_url")
    model = _litellm_model(model_name)
    # OpenHands 1.49.6 defaults this to "high". Reasoning levels are not
    # universal across gateways, so always use the run-frozen explicit value.
    timeout_seconds = max(1, math.ceil(float(request.get("llm_timeout_seconds") or 300)))
    llm_kwargs = {"model": model, "base_url": base_url,
                  "max_output_tokens": 8192, "api_key": SecretStr(key),
                  "usage_id": "medbench_coding_worker",
                  # Let the stage supervisor own the retry budget.  OpenHands'
                  # SDK default (five retries) can otherwise hide a bad endpoint
                  # for most of the stage timeout.
                  "num_retries": int(request.get("llm_max_retries") or 0),
                  "timeout": timeout_seconds,
                  }
    # OpenHands defaults to high, which is not accepted by this vLLM service;
    # keep the previously working explicit medium default.
    llm_kwargs["reasoning_effort"] = request.get("reasoning_effort") or "medium"
    llm = LLM(**llm_kwargs)
    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)])
    event_path = Path(request["construction_events_path"]) if request.get("construction_events_path") else None
    if event_path:
        event_path.parent.mkdir(parents=True, exist_ok=True)

    progress = {"iterations": 0, "model_turns": 0, "tool_calls": 0,
                "events": 0, "tool_errors": 0, "empty_responses": 0,
                "no_progress": 0, "last_action": "", "artifact_digest": ""}
    guard = request.get("progress_guard") or {}
    previous_digest = ""

    def artifact_digest() -> str:
        digest = hashlib.sha256()
        # Runtime state, heartbeats and forwarding offsets describe liveness;
        # they must not make a stalled task look like useful output progress.
        for root_name in ("outputs",):
            root = workspace / root_name
            if not root.is_dir():
                continue
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                digest.update(path.relative_to(workspace).as_posix().encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def on_event(event):
        nonlocal previous_digest
        event_name = type(event).__name__
        progress["events"] += 1
        if event_name == "ActionEvent":
            progress["tool_calls"] += 1
            # A tool call is observable work even when it only inspects the
            # source.  The output digest intentionally excludes read-only
            # work, so reset the stalled-output counter here.
            progress["no_progress"] = 0
            action = getattr(event, "action", None)
            tool_name = str(getattr(action, "name", getattr(action, "action", "")))
            arguments = getattr(action, "arguments", getattr(action, "args", {})) or {}
            normalized, stripped = normalize_tool_arguments(tool_name, arguments)
            # OpenHands executes the action object after callbacks.  Mutate the
            # actual object when the SDK exposes a mutable arguments field so
            # harmless gateway metadata cannot reach the executor.
            if stripped and action is not None:
                for attribute in ("arguments", "args"):
                    if hasattr(action, attribute):
                        try:
                            setattr(action, attribute, normalized)
                            break
                        except Exception:
                            continue
            command = normalized.get("command", "") if isinstance(normalized, dict) else ""
            if validate_terminal_command(command):
                raise RuntimeError("Preparation stage guard: blocked_command")
            if stripped:
                progress["tool_metadata_stripped"] = progress.get("tool_metadata_stripped", 0) + len(stripped)
        if event_name in {"AgentEvent", "AgentFinishEvent", "MessageEvent"}:
            progress["model_turns"] += 1
            progress["iterations"] = progress["model_turns"]
        if event_name in {"ObservationEvent", "AgentErrorEvent"}:
            progress["last_action"] = event_name
            if "Error" in event_name:
                progress["tool_errors"] += 1
            current_digest = artifact_digest()
            if current_digest == previous_digest:
                progress["no_progress"] += 1
            else:
                progress["no_progress"] = 0
                previous_digest = current_digest
            progress["artifact_digest"] = current_digest
            if progress["tool_errors"] >= int(guard.get("tool_error_limit", 3)):
                raise RuntimeError("Preparation stage guard: tool_error_budget")
            if progress["no_progress"] >= int(guard.get("no_progress_limit", 8)):
                raise RuntimeError("Preparation stage guard: no_progress")
        _write_session_state(workspace, request, status="running", progress=progress)
        if request.get("construction_events_path"):
            item = {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "coding_worker_event", "run_id": request["run_id"],
                    "agent": request.get("agent", "DatasetProcessingAgent"), "component": "OpenHandsCodingWorker",
                    "dataset_id": request["dataset_id"],
                     "source_id": request["source_id"], "phase": request.get("stage", "prepare"),
                    "attempt": request.get("attempt"), "conversation_id": request["conversation_id"],
                    "model": request.get("model"), "reasoning_effort": request.get("reasoning_effort") or None,
                     "operation": type(event).__name__,
                     "status": "running",
                     "tool_name": tool_name if event_name == "ActionEvent" else "",
                     "command": command if event_name == "ActionEvent" else "",
                     "progress": {"model_turns": progress["model_turns"], "tool_calls": progress["tool_calls"],
                                  "tool_errors": progress["tool_errors"], "artifact_digest": progress["artifact_digest"]}}
            with event_path.open("a", encoding="utf-8") as events:
                events.write(json.dumps(item, ensure_ascii=False) + "\n")

    conversation = Conversation(agent=agent, workspace=workspace,
                                persistence_dir=workspace / "conversations",
                                conversation_id=__import__("uuid").UUID(request["conversation_id"]),
                                max_iteration_per_run=request["max_iterations"], stuck_detection=True,
                                delete_on_close=False, callbacks=[on_event])
    prompt_path = Path(request.get("prompt_path") or "")
    if not prompt_path.is_file():
        raise RuntimeError(f"Worker prompt is missing: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    if not request.get("resume_existing"):
        conversation.send_message(prompt)
    try:
        conversation.run()
    except Exception as exc:
        # Keep the process contract machine-readable while leaving the full
        # traceback in stderr for diagnosis.
        guarded = str(exc).startswith("Preparation stage guard:")
        reason = str(exc).split(":", 1)[1].strip() if guarded else "worker_error"
        result = {"stage": request["stage"], "attempt": request.get("attempt", 1),
                  "conversation_id": request["conversation_id"], "execution_key": request["execution_key"],
                  "status": "blocked" if guarded else "failed", "outputs": {}, "question_refs": [],
                  "diagnostics": [], "diagnostic": f"OpenHands conversation failed: {type(exc).__name__}: {exc}",
                  "termination_reason": reason, "progress": progress,
                  "input_digest": request.get("input_digest", ""),
                  "token_count": request.get("prompt_token_count")}
        _write_session_state(workspace, request, status=result["status"], summary=result["diagnostic"],
                             termination_reason=reason, progress=progress)
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        if guarded:
            return
        raise
    diagnostic = replay_source_script(request, workspace) if request.get("stage") == "understand" and (workspace / "outputs" / "inspect_source.py").is_file() else ""
    produced = {}
    allowed = set(request.get("allowed_outputs", [])) | set(request.get("required_outputs", []))
    output_root = workspace / "outputs"
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        name = path.relative_to(output_root).as_posix()
        # Every extra file is admissible only as a worker output under the
        # declared output root; output_manifest.json records its dependency
        # role and host validation decides whether it is publishable.
        if name in allowed or "output_manifest.json" in allowed or path.name not in {"complete.json", "preparation_report.json", "quarantine.jsonl"}:
            produced[name] = str(path.resolve())
    submission_path = workspace / "outputs" / "submission.json"
    submission = None
    if submission_path.is_file():
        try:
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostic = f"Invalid submission.json: {exc}"
    else:
        diagnostic = diagnostic or "Required worker output is missing: submission.json"
    complete = not diagnostic and set(request["required_outputs"]) <= set(produced)
    requested = submission.get("requested_outcome") if isinstance(submission, dict) else "blocked"
    result_status = "submitted" if complete and requested == "submitted" else ("needs_input" if requested == "needs_input" else "blocked" if requested == "blocked" else "failed")
    result = {"stage": request["stage"], "attempt": request.get("attempt", 1),
              "conversation_id": request["conversation_id"], "execution_key": request["execution_key"],
              "status": result_status, "outputs": produced,
              "question_refs": submission.get("question_refs", []) if isinstance(submission, dict) else [],
              "diagnostics": [],
              "diagnostic": "" if complete else diagnostic or (submission.get("summary", "") if isinstance(submission, dict) else "Required worker outputs are missing"),
              "termination_reason": "completed" if result_status == "submitted" else "output_invalid",
              "progress": progress, "input_digest": request.get("input_digest", ""),
              "submission_path": "/workspace/outputs/submission.json",
              "token_count": request.get("prompt_token_count")}
    _write_session_state(workspace, request,
                         status=result["status"], summary=result["diagnostic"],
                         termination_reason=result["termination_reason"], progress=progress)
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

