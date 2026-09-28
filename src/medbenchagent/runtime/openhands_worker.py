"""OpenHands SDK process entrypoint for one resumable coding task."""

from __future__ import annotations

import json
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _litellm_model(model: str) -> str:
    """Return the provider-qualified model name required by LiteLLM."""
    return model if "/" in model else "openai/" + model


def _workflow_skill() -> str:
    """Load the repository skill so runtime and the reusable skill stay aligned."""
    path = Path(__file__).resolve().parents[3] / ".codex" / "skills" / "medbench-data-preparation" / "SKILL.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_session_state(workspace: Path, request: dict, *, status: str, summary: str = "") -> None:
    state = {
        "schema_version": "preparation-session.v1",
        "task_id": request["task_id"],
        "kind": request["kind"],
        "source_root": request["source_root"],
        "conversation_id": request["conversation_id"],
        "attempt": request.get("attempt", 1),
        "status": status,
        "summary": summary[-4000:],
        "required_outputs": request.get("required_outputs", []),
    }
    (workspace / "state").mkdir(parents=True, exist_ok=True)
    (workspace / "state" / "session_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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

    def on_event(event):
        if request.get("construction_events_path"):
            item = {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "coding_worker_event", "run_id": request["run_id"],
                    "agent": "OpenHandsCodingWorker", "dataset_id": request["dataset_id"],
                    "source_id": request["source_id"], "phase": "CandidateDiscovery",
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
    prompt = _workflow_skill() + "\n\n" + (
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
        f"Objective: {request['objective']}\nRequired files under ./outputs: {outputs}. "
    )
    if request["kind"] in {"preparation", "source_inventory"}:
        prompt += ("Inspect the complete relevant population, not a sample. Use source-relative paths "
                   "in reports. State unresolved semantic questions in source_report.json uncertainties. "
                   "Run outputs/inspect_source.py and check its results before finishing.")
    elif request["kind"] == "dataset_adapter":
        prompt += ("Read inputs/adapter_contract.txt and inputs/spec.json. Test adapter.py using "
                   "MEDBENCH_SOURCE_ROOT=$MEDBENCH_SOURCE_ROOT, MEDBENCH_SPEC_PATH=./inputs/spec.json, "
                   "MEDBENCH_INPUTS_PATH=./inputs/adapter_inputs.json and a writable trial "
                   "MEDBENCH_WORK_DIR. Keep trial outputs under ./work.")
    if request["feedback"]:
        prompt += "\nValidator feedback to fix: " + request["feedback"]
    conversation.send_message(prompt)
    try:
        conversation.run()
    except Exception as exc:
        # Keep the process contract machine-readable while leaving the full
        # traceback in stderr for diagnosis.
        result = {"status": "failed", "conversation_id": request["conversation_id"],
                  "outputs": {}, "question": "",
                  "diagnostic": f"OpenHands conversation failed: {type(exc).__name__}: {exc}"}
        _write_session_state(workspace, request, status="blocked", summary=result["diagnostic"])
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        raise
    diagnostic = replay_source_script(request, workspace) if request["kind"] in {"preparation", "source_inventory"} else ""
    produced = {name: str((workspace / "outputs" / name).resolve()) for name in request["required_outputs"]
                if (workspace / "outputs" / name).is_file()}
    complete = not diagnostic and len(produced) == len(request["required_outputs"])
    result = {"status": "complete" if complete else "failed",
              "conversation_id": request["conversation_id"], "outputs": produced,
              "question": "", "diagnostic": "" if complete else diagnostic or "Required worker outputs are missing",
              "state": "submitted" if complete else "validation_failed"}
    _write_session_state(workspace, request,
                         status=result["state"], summary=result["diagnostic"])
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
