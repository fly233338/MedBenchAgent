"""Construction of the single, complete Coding Session task."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from medbenchagent.settings import get_settings


def build_data_preparation_task(context: dict, workspace: Path, *, feedback: str = ""):
    from medbenchagent.runtime.dataset_runtime import DatasetAgentTask
    from medbenchagent.runtime.preparation_session import load_session

    sources = context.get("sources", [])
    if not sources:
        raise ValueError("Dataset preparation requires at least one source")
    primary = context.get("active_source") or sources[0]
    session = load_session(workspace)
    prompt = (Path(__file__).parents[1] / "prompts" / "data_preparation_agent.md").read_text(encoding="utf-8")
    task_context = {"sources": sources, "source_manifests": context.get("source_manifests", {}),
                    "schema": str(Path(__file__).parents[1] / "schemas" / "annotation.v5.schema.json"),
                    "workspace": str(workspace), "validator": "medbenchagent.data.v5_validation",
                    "feedback": feedback}
    objective = prompt + "\n\nFrozen task context:\n" + json.dumps(task_context, ensure_ascii=False)
    if feedback:
        objective += "\n\nValidator feedback to repair:\n" + feedback[:12000]
    conversation_id = session.get("conversation_id") or str(uuid.uuid5(uuid.NAMESPACE_URL,
                                                                        f"{context['run_id']}:{context['dataset_scope']}:preparation"))
    return DatasetAgentTask(
        task_id=f"{context['run_id']}:{context['dataset_scope']}:preparation", kind="preparation",
        source_root=str(Path(primary["root"]).resolve()),
        source_manifest=context.get("source_manifests", {}).get(primary["id"], context["source_manifest"]),
        source_roots={item["id"]: item["root"] for item in sources},
        workspace=str(workspace), objective=objective,
         required_outputs=["annotation.jsonl", "asset_manifest.jsonl", "provenance.jsonl", "preparation_report.json", "conversion_config.json", "grouping_policy.json", "mapping.json", "conversion_version", "complete.json", "scripts/prepare.py", "notes/inventory.md", "notes/mapping.md", "notes/dataset_notes.md"],
        feedback=feedback, attempt=session.get("attempt", 1), conversation_id=conversation_id,
        max_iterations=context.get("coding_worker_max_iterations", 80), timeout_seconds=context.get("coding_worker_timeout_seconds", 3600),
        model=get_settings().llm.model, base_url=get_settings().llm.base_url,
        reasoning_effort=get_settings().llm.reasoning_effort, run_id=context["run_id"],
        dataset_id=context["dataset_scope"], source_id=primary["id"], workflow_step="preparation")
