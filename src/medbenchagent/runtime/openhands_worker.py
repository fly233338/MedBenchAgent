"""OpenHands SDK process entrypoint for one preparation attempt."""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def normalize_tool_arguments(tool_name: str, arguments: dict) -> tuple[dict, list[str]]:
    normalized = dict(arguments or {})
    stripped = []
    if tool_name in {"terminal", "file_editor"}:
        for key in ("description_note", "description"):
            if key in normalized:
                normalized.pop(key, None)
                stripped.append(key)
    return normalized, stripped


def validate_terminal_command(command: str) -> str:
    lowered = (command or "").casefold()
    forbidden = ("pip install", "python -m pip", "uv pip", "curl ", "wget ", "invoke-webrequest")
    if any(token in lowered for token in forbidden):
        return "Package installation and unrelated network access are disabled in preparation workers"
    return ""


def typed_action_payload(event) -> tuple[str, dict, str]:
    tool_name = str(getattr(event, "tool_name", "") or "")
    action = getattr(event, "action", None)
    if not tool_name or action is None:
        raise ValueError("ActionEvent is missing typed tool_name/action fields")
    if tool_name == "terminal":
        command = getattr(action, "command", None)
        if not isinstance(command, str):
            raise ValueError("terminal ActionEvent.action.command must be a string")
        return tool_name, {"command": command}, command
    if hasattr(action, "model_dump"):
        arguments = action.model_dump(mode="json")
    elif isinstance(action, dict):
        arguments = dict(action)
    else:
        raise ValueError(f"{tool_name} ActionEvent has no typed argument payload")
    if not isinstance(arguments, dict):
        raise ValueError(f"{tool_name} ActionEvent arguments must be an object")
    return tool_name, arguments, ""


def serializable_tool_call(value):
    """Keep malformed tool calls observable without letting logging fail."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except (TypeError, ValueError):
            pass
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def observation_failure(event) -> str:
    observation = getattr(event, "observation", None)
    if observation is None:
        return ""
    if bool(getattr(observation, "is_error", False)) or bool(getattr(observation, "error", False)):
        return "tool_observation_error"
    exit_code = getattr(observation, "exit_code", None)
    if isinstance(exit_code, int) and exit_code not in {0, -1}:
        return f"command_exit_{exit_code}"
    return ""


def _litellm_model(model: str) -> str:
    return model if "/" in model else "openai/" + model


def main(request_path: str, result_path: str) -> None:
    from pydantic import SecretStr
    from openhands.sdk import Agent, Conversation, LLM, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    workspace = Path(request["workspace"]).resolve()
    key = os.getenv("MEDBENCH_LLM_API_KEY")
    if not key:
        raise RuntimeError("Coding worker requires the configured LLM API key")
    os.environ.pop("MEDBENCH_LLM_API_KEY", None)
    model_name = str(request.get("model") or "").strip()
    base_url = str(request.get("base_url") or "").strip()
    if not model_name or not base_url:
        raise RuntimeError("Coding worker requires a non-empty run-frozen model and base_url")
    llm = LLM(
        model=_litellm_model(model_name),
        base_url=base_url,
        max_output_tokens=int(request.get("budget", {}).get("output_reserve_tokens", 4096)),
        api_key=SecretStr(key),
        usage_id="medbench_coding_worker",
        num_retries=int(request.get("llm_max_retries") or 0),
        timeout=max(1, math.ceil(float(request.get("llm_timeout_seconds") or 300))),
        reasoning_effort=request.get("reasoning_effort") or "medium",
    )
    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)])
    event_path = Path(request["construction_events_path"]) if request.get("construction_events_path") else None
    if event_path:
        event_path.parent.mkdir(parents=True, exist_ok=True)

    progress = {
        "iterations": 0, "model_turns": 0, "tool_calls": 0, "events": 0,
        "tool_errors": 0, "recovered_tool_errors": 0,
    }

    def on_event(event):
        event_name = type(event).__name__
        progress["events"] += 1
        tool_name = ""
        command = ""
        non_executable = False
        tool_call = None
        event_error = ""
        if event_name == "ActionEvent":
            progress["tool_calls"] += 1
            action = getattr(event, "action", None)
            tool_name = str(getattr(event, "tool_name", "") or "")
            if action is None:
                # OpenHands emits this valid event before AgentErrorEvent when a
                # tool call cannot be normalized or validated. Let the SDK
                # return the error to the model instead of aborting the run.
                non_executable = True
                progress["non_executable_tool_calls"] = progress.get("non_executable_tool_calls", 0) + 1
                progress["tool_errors"] += 1
                progress["recovered_tool_errors"] += 1
                event_error = "tool_action_validation: missing typed action"
                tool_call = serializable_tool_call(getattr(event, "tool_call", None))
            else:
                try:
                    tool_name, arguments, command = typed_action_payload(event)
                    arguments, stripped = normalize_tool_arguments(tool_name, arguments)
                    if validate_terminal_command(command):
                        raise RuntimeError("Preparation stage guard: blocked_command")
                    if stripped:
                        progress["tool_metadata_stripped"] = progress.get("tool_metadata_stripped", 0) + len(stripped)
                except RuntimeError:
                    raise
                except (TypeError, ValueError) as exc:
                    non_executable = True
                    progress["tool_errors"] += 1
                    progress["recovered_tool_errors"] += 1
                    event_error = f"tool_action_validation: {exc}"
        if event_name in {"AgentEvent", "AgentFinishEvent", "MessageEvent"}:
            progress["model_turns"] += 1
            progress["iterations"] = progress["model_turns"]
        if event_name in {"ObservationEvent", "AgentErrorEvent"}:
            if event_name == "AgentErrorEvent" or observation_failure(event):
                progress["tool_errors"] += 1
                if event_name == "ObservationEvent":
                    progress["recovered_tool_errors"] += 1
        if event_path:
            item = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "coding_worker_event",
                "run_id": request["run_id"],
                "agent": request.get("agent", "DatasetProcessingAgent"),
                "component": "OpenHandsCodingWorker",
                "dataset_id": request["dataset_id"],
                "phase": request.get("stage", "prepare"),
                "attempt": request.get("attempt"),
                "attempt_id": request.get("attempt_id", ""),
                "conversation_id": request["conversation_id"],
                "model": request.get("model"),
                "reasoning_effort": request.get("reasoning_effort") or None,
                "operation": event_name,
                "status": "running",
                "tool_name": tool_name,
                "command": command,
                "non_executable": non_executable,
                "tool_call": tool_call,
                "error": event_error or str(getattr(event, "error", "") or ""),
                "progress": dict(progress),
            }
            with event_path.open("a", encoding="utf-8") as events:
                events.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        persistence_dir=workspace / "work" / "conversations",
        conversation_id=__import__("uuid").UUID(request["conversation_id"]),
        max_iteration_per_run=request["budget"]["max_iterations"],
        stuck_detection=True,
        delete_on_close=False,
        callbacks=[on_event],
    )
    prompt_path = Path(request.get("prompt_path") or "")
    if not prompt_path.is_file():
        raise RuntimeError(f"Worker prompt is missing: {prompt_path}")
    if not request.get("resume_existing"):
        conversation.send_message(prompt_path.read_text(encoding="utf-8"))
    try:
        conversation.run()
    except Exception as exc:
        guarded = str(exc).startswith("Preparation stage guard:")
        reason = str(exc).split(":", 1)[1].strip() if guarded else "worker_error"
        result = {
            "stage": request["stage"], "attempt": request.get("attempt", 1),
            "attempt_id": request.get("attempt_id", ""),
            "conversation_id": request["conversation_id"], "execution_key": request["execution_key"],
            "status": "blocked" if guarded else "failed", "outputs": {}, "question_refs": [],
            "diagnostics": [], "diagnostic": f"OpenHands conversation failed: {type(exc).__name__}: {exc}",
            "termination_reason": reason, "progress": progress,
            "input_digest": request.get("input_digest", ""),
            "token_count": request.get("prompt_token_count"),
        }
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        if guarded:
            return
        raise

    produced = {}
    allowed = set(request.get("allowed_outputs", [])) | set(request.get("required_outputs", []))
    output_root = workspace / "work" / "outputs"
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        name = path.relative_to(output_root).as_posix()
        if name in allowed or "output_manifest.json" in allowed or path.name not in {"complete.json", "preparation_report.json", "quarantine.jsonl"}:
            produced[name] = str(path.resolve())
    submission_path = workspace / "work" / "outputs" / "submission.json"
    submission = None
    diagnostic = ""
    if submission_path.is_file():
        try:
            submission = json.loads(submission_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            diagnostic = f"Invalid submission.json: {exc}"
    else:
        diagnostic = "Required worker output is missing: submission.json"
    if not diagnostic and submission and submission.get("requested_outcome") == "submitted":
        validation_path = workspace / "work" / "validation.json"
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if validation.get("status") != "accepted":
                diagnostic = "Shared stage validator did not accept the candidate"
        except (OSError, json.JSONDecodeError, AttributeError):
            diagnostic = "Shared stage validator result is missing or invalid: work/validation.json"
    complete = not diagnostic and set(request["required_outputs"]) <= set(produced)
    requested = submission.get("requested_outcome") if isinstance(submission, dict) else "blocked"
    result_status = "submitted" if complete and requested == "submitted" else ("needs_input" if requested == "needs_input" else "blocked" if requested == "blocked" else "failed")
    result = {
        "stage": request["stage"], "attempt": request.get("attempt", 1),
        "attempt_id": request.get("attempt_id", ""),
        "conversation_id": request["conversation_id"], "execution_key": request["execution_key"],
        "status": result_status, "outputs": produced,
        "question_refs": submission.get("question_refs", []) if isinstance(submission, dict) else [],
        "diagnostics": [], "diagnostic": "" if complete else diagnostic or submission.get("summary", ""),
        "termination_reason": "completed" if result_status == "submitted" else "output_invalid",
        "progress": progress, "input_digest": request.get("input_digest", ""),
        "submission_path": "/workspace/work/outputs/submission.json",
        "token_count": request.get("prompt_token_count"),
    }
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

