"""Compatibility coding-worker facade used by source inventory and adapters."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CodingTask(BaseModel):
    task_id: str
    kind: Literal["source_inventory", "dataset_adapter", "preparation"]
    source_root: str
    source_manifest: str
    workspace: str
    objective: str
    required_outputs: list[str]
    model: str
    base_url: str
    feedback: str = ""
    attempt: int = Field(default=1, ge=1)
    max_iterations: int = 40
    timeout_seconds: int = 1800
    conversation_id: str | None = None
    run_id: str = ""
    dataset_id: str = ""
    source_id: str = ""


class CodingResult(BaseModel):
    status: Literal["complete", "needs_input", "blocked", "failed"]
    conversation_id: str
    outputs: dict[str, str] = {}
    question: str = ""
    diagnostic: str = ""


def _copy_context_summary(path: str, workspace: Path) -> None:
    source = Path(path).resolve()
    target = (workspace / "state" / source.name).resolve()
    if source != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


class OpenHandsCodingWorker:
    def run(self, task: CodingTask) -> CodingResult:
        workspace = Path(task.workspace).resolve()
        for name in ("inputs", "work", "outputs", "state", "logs", "conversations"):
            (workspace / name).mkdir(parents=True, exist_ok=True)
        conversation_id = task.conversation_id or str(uuid.uuid5(uuid.NAMESPACE_URL, task.task_id + f":{task.attempt}"))
        request = task.model_dump()
        request.pop("source_manifest", None)
        request.update(workspace="/workspace", source_root="/workspace/source",
                       conversation_id=conversation_id)
        request_path = workspace / "state" / "coding_task.json"
        result_path = workspace / "state" / f"coding_result_{task.kind}.json"
        if request_path.is_file() and result_path.is_file():
            try:
                previous = json.loads(request_path.read_text(encoding="utf-8"))
                if previous == request and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                    return CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = ["docker", "run", "--rm", "--pull=never", "--mount",
                   f"type=bind,src={Path(task.source_root).resolve()},dst=/workspace/source,readonly",
                   "-w", "/workspace", "medbench-openhands:local", "python",
                   "/app/openhands_worker.py", "/workspace/state/coding_task.json",
                   f"/workspace/state/coding_result_{task.kind}.json"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=task.timeout_seconds, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CodingResult(status="blocked", conversation_id=conversation_id, diagnostic=str(exc))
        if completed.returncode or not result_path.is_file():
            return CodingResult(status="failed", conversation_id=conversation_id,
                                diagnostic=(completed.stderr or "worker failed")[-4000:])
        return CodingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
