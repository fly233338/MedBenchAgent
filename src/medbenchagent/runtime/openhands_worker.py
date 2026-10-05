"""OpenHands SDK process entrypoint for one resumable coding task."""

from __future__ import annotations

import json
import filecmp
import hashlib
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
    state = {"schema_version": "preparation-session", "task_id": request["task_id"],
             "kind": request["kind"], "source_root": request["source_root"],
             "source_digest": request.get("source_digest", ""),
             "workflow_step": request.get("workflow_step", ""),
             "stage": request.get("stage", request.get("workflow_step", "candidate")),
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
    replay_marker = workspace / "state" / "source_inventory_replayed.json"
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
    model = _litellm_model(request["model"])
    # OpenHands 1.49.6 defaults this to "high". Reasoning levels are not
    # universal across gateways, so always use the run-frozen explicit value.
    llm_kwargs = {"model": model, "base_url": request["base_url"],
                  "max_output_tokens": 8192, "api_key": SecretStr(key),
                  "usage_id": "medbench_coding_worker"}
    # OpenHands defaults to high, which is not accepted by this vLLM service;
    # keep the previously working explicit medium default.
    llm_kwargs["reasoning_effort"] = request.get("reasoning_effort") or "medium"
    llm = LLM(**llm_kwargs)
    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)])
    event_path = Path(request["construction_events_path"]) if request.get("construction_events_path") else None
    if event_path:
        event_path.parent.mkdir(parents=True, exist_ok=True)

    progress = {"iterations": 0, "tool_errors": 0, "empty_responses": 0,
                "no_progress": 0, "last_action": "", "artifact_digest": ""}
    guard = request.get("progress_guard") or {}
    previous_digest = ""

    def artifact_digest() -> str:
        digest = hashlib.sha256()
        for root_name in ("state", "outputs"):
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
        if event_name == "ActionEvent":
            action = getattr(event, "action", None)
            tool_name = str(getattr(action, "name", getattr(action, "action", "")))
            arguments = getattr(action, "arguments", getattr(action, "args", {})) or {}
            normalized, stripped = normalize_tool_arguments(tool_name, arguments)
            command = normalized.get("command", "") if isinstance(normalized, dict) else ""
            if validate_terminal_command(command):
                raise RuntimeError("Preparation stage guard: blocked_command")
            if stripped:
                progress["tool_metadata_stripped"] = progress.get("tool_metadata_stripped", 0) + len(stripped)
        if event_name in {"ActionEvent", "ObservationEvent", "AgentErrorEvent"}:
            progress["iterations"] += 1
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
                    "source_id": request["source_id"], "phase": request.get("stage", "candidate"),
                    "attempt": request.get("attempt"), "conversation_id": request["conversation_id"],
                    "model": request.get("model"), "reasoning_effort": request.get("reasoning_effort") or None,
                    "operation": type(event).__name__,
                    "status": "running"}
            with event_path.open("a", encoding="utf-8") as events:
                events.write(json.dumps(item, ensure_ascii=False) + "\n")

    conversation = Conversation(agent=agent, workspace=workspace,
                                persistence_dir=workspace / "conversations",
                                conversation_id=__import__("uuid").UUID(request["conversation_id"]),
                                max_iteration_per_run=request["max_iterations"], stuck_detection=True,
                                delete_on_close=False, callbacks=[on_event])
    outputs = ", ".join(request["required_outputs"])
    stage = request.get("stage", "candidate")
    stage_prompts = {
        "inventory": "Inventory the complete source population and write only the declared inventory outputs. Do not design mappings or adapters.",
        "semantic": "Use the supplied inventory and source evidence to write a source-grounded mapping proposal. Record unresolved semantic questions instead of guessing.",
        "adapter": "Implement and test a deterministic adapter from the frozen semantic context. Keep trial outputs under work and do not publish candidate data.",
        "candidate": "Run the deterministic adapter over the complete population and write the declared core candidate outputs incrementally.",
        "repair": "Read the bounded validator feedback and repair only the responsible adapter or mapping. Re-run the affected checks and write the declared outputs.",
    }
    context_text = "{}"
    if request.get("context_path"):
        context_file = Path(request["context_path"])
        if context_file.is_file():
            context_text = context_file.read_text(encoding="utf-8")[:24000]
    prompt = (
        "You are a bounded MedBench preparation stage agent. Work only inside this workspace. "
        "You are the MedBench preparation coding agent. Work only inside this workspace. "
        f"The read-only source is at {request['source_root']} and additional inputs are in ./inputs. "
        "Write scratch files under ./work, persistent progress under ./state, and candidate files under ./outputs. "
        "The source is immutable: never edit, copy-over, delete, or rename source files. "
        "Use only source files, frozen input documents, user clarifications, and tool results as evidence. "
        "Never infer medical meaning, identity, labels, missingness, or population membership from general knowledge. "
        "Preserve raw strings and distinguish missing from negative. Every image, mask, finding, and label must be "
        "traceable to a source-relative locator. Cover the complete source population; if completeness cannot be "
        "proved, record that explicitly and ask one concrete question naming the field/value, context, and consequence. "
        "You may write and run a deterministic Python adapter for non-standard structures such as nested arrays or "
        "ampersand-separated mask cells, but its output is only a candidate until MedBench validates it. "
        "Before finishing, run the script, inspect counts and outputs, fix failures, and run it again. "
        "Do not repeat an inspection whose result is already present in workspace/state or the supplied feedback. "
        "Do not claim publication or readiness; submit candidates and an explicit summary of unresolved questions. "
        f"Stage: {stage}. Stage objective: {stage_prompts.get(stage, request['objective'])}\n"
        f"Host context (authoritative references only): {context_text}\n"
        f"Required files under ./outputs: {outputs}. "
    )
    if stage == "inventory":
        prompt += " Inspect the complete population, use source-relative paths, and run the inspection script before finishing."
    elif stage == "adapter":
        prompt += " Test the adapter against representative and boundary records before finishing."
    if request["feedback"]:
        prompt += "\nValidator feedback to fix (full report is in workspace): " + request["feedback"][:6000]
    conversation.send_message(prompt)
    try:
        conversation.run()
    except Exception as exc:
        # Keep the process contract machine-readable while leaving the full
        # traceback in stderr for diagnosis.
        guarded = str(exc).startswith("Preparation stage guard:")
        reason = str(exc).split(":", 1)[1].strip() if guarded else "worker_error"
        result = {"status": "blocked" if guarded else "failed", "conversation_id": request["conversation_id"],
                  "outputs": {}, "question": "",
                  "diagnostic": f"OpenHands conversation failed: {type(exc).__name__}: {exc}",
                  "termination_reason": reason, "progress": progress}
        _write_session_state(workspace, request, status=result["status"], summary=result["diagnostic"],
                             termination_reason=reason, progress=progress)
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        if guarded:
            return
        raise
    diagnostic = replay_source_script(request, workspace) if request.get("stage") == "inventory" else ""
    produced = {name: str((workspace / "outputs" / name).resolve()) for name in request["required_outputs"]
                if (workspace / "outputs" / name).is_file()}
    complete = not diagnostic and len(produced) == len(request["required_outputs"])
    result = {"status": "complete" if complete else "failed",
              "conversation_id": request["conversation_id"], "outputs": produced,
              "question": "", "diagnostic": "" if complete else diagnostic or "Required worker outputs are missing",
              "state": "submitted" if complete else "validation_failed",
              "termination_reason": "completed" if complete else "output_invalid",
              "progress": progress}
    _write_session_state(workspace, request,
                         status=result["state"], summary=result["diagnostic"],
                         termination_reason=result["termination_reason"], progress=progress)
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

