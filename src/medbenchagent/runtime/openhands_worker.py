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


def replay_source_script(request: dict, workspace: Path) -> str:
    """Re-run the worker's inspection code twice before accepting its files."""
    script = workspace / "outputs" / "inspect_source.py"
    if not script.is_file():
        return "Required inspection script is missing"
    replay_marker = workspace / "state" / "source_inventory_replayed.json"
    replay_marker.unlink(missing_ok=True)
    replay = []
    with tempfile.TemporaryDirectory(dir=workspace / "work") as first, \
            tempfile.TemporaryDirectory(dir=workspace / "work") as second:
        for trial in (Path(first), Path(second)):
            environment = {**os.environ, "MEDBENCH_SOURCE_ROOT": str(workspace / "source"),
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
    key = os.getenv("MEDBENCH_LLM_API_KEY")
    if not key:
        raise RuntimeError("Coding worker requires the configured LLM API key")
    # Keep the credential in the LLM object, away from terminal/file-editor subprocesses.
    os.environ.pop("MEDBENCH_LLM_API_KEY", None)
    # The run freezes the exact gateway model id.  LiteLLM needs an explicit
    # provider at this adapter boundary; the gateway still receives the raw id.
    model = _litellm_model(request["model"])
    llm = LLM(model=model, base_url=request["base_url"], max_output_tokens=8192,
              api_key=SecretStr(key), usage_id="medbench_coding_worker")
    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)])
    event_path = Path(request["construction_events_path"]) if request.get("construction_events_path") else None
    if event_path:
        event_path.parent.mkdir(parents=True, exist_ok=True)

    def on_event(event):
        if request.get("construction_events_path"):
            item = {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "coding_worker_event", "run_id": request["run_id"],
                    "agent": "OpenHandsCodingWorker", "dataset_id": request["dataset_id"],
                    "source_id": request["source_id"], "operation": type(event).__name__,
                    "status": "running"}
            with event_path.open("a", encoding="utf-8") as events:
                events.write(json.dumps(item, ensure_ascii=False) + "\n")

    conversation = Conversation(agent=agent, workspace=workspace,
                                persistence_dir=workspace / "conversations",
                                conversation_id=__import__("uuid").UUID(request["conversation_id"]),
                                max_iteration_per_run=request["max_iterations"], stuck_detection=True,
                                delete_on_close=False, callbacks=[on_event])
    outputs = ", ".join(request["required_outputs"])
    prompt = (
        "You are a data inspection and Python coding worker. Work only inside this workspace. "
        "The read-only source is in ./source and additional inputs are in ./inputs. "
        "Write scratch files under ./work and final candidate files under ./outputs. "
        "Do not infer clinical meaning, publish data, or edit the source. "
        f"Objective: {request['objective']}\n"
        f"Required files under ./outputs: {outputs}. "
    )
    if request["kind"] == "source_inventory":
        prompt += ("Inspect the complete relevant population, not a sample. Use source-relative paths "
                   "in reports. State unresolved semantic questions in source_report.json uncertainties. "
                   "Run outputs/inspect_source.py and check its results before finishing.")
    else:
        prompt += ("Read inputs/adapter_contract.txt and inputs/spec.json. Test adapter.py using "
                   "MEDBENCH_SOURCE_ROOT=./source, MEDBENCH_SPEC_PATH=./inputs/spec.json, "
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
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        raise
    diagnostic = replay_source_script(request, workspace) if request["kind"] == "source_inventory" else ""
    produced = {name: str((workspace / "outputs" / name).resolve()) for name in request["required_outputs"]
                if (workspace / "outputs" / name).is_file()}
    complete = not diagnostic and len(produced) == len(request["required_outputs"])
    result = {"status": "complete" if complete else "failed",
              "conversation_id": request["conversation_id"], "outputs": produced,
              "question": "", "diagnostic": "" if complete else diagnostic or "Required worker outputs are missing"}
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
