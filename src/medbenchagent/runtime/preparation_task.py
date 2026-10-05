"""Construction of the single, complete Coding Session task."""

from __future__ import annotations

import uuid
from pathlib import Path

from medbenchagent.settings import get_settings


STAGE_OUTPUTS = {
    "inventory": ["source_report.json", "image_population.jsonl", "inspect_source.py"],
    "semantic": ["mapping.json", "semantic_questions.json"],
    "adapter": ["adapter.py"],
    "candidate": [
        "cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl", "conversion_config.json",
        "grouping_policy.json", "mapping.json", "scripts/prepare.py",
    ],
    "repair": [
        "cases.jsonl", "asset_manifest.jsonl", "provenance.jsonl", "conversion_config.json",
        "grouping_policy.json", "mapping.json", "scripts/prepare.py",
    ],
}


def build_data_preparation_task(context: dict, workspace: Path, *, feedback: str = "",
                                stage: str = "candidate", attempt: int | None = None):
    from medbenchagent.runtime.dataset_runtime import DatasetAgentTask
    sources = context.get("sources", [])
    if not sources:
        raise ValueError("Dataset preparation requires at least one source")
    primary = context.get("active_source") or sources[0]
    prompt = (Path(__file__).parents[1] / "prompts" / "data_preparation_agent.md").read_text(encoding="utf-8")
    objective = prompt
    objective += (
        "\n\nOutput boundary: the original source is read-only. Do not ask where to write the processed dataset. "
        "Write every candidate file only below ./outputs (including ./outputs/scripts and ./outputs/notes). "
        "The host will validate the candidate and publish it to the prepared bundle path recorded as "
        "prepared_bundle_root; the worker must never write to that path directly."
    )
    if feedback:
        objective += "\n\nValidator feedback to repair:\n" + feedback[:12000]
    stage_attempt = attempt or int(context.get("preparation_attempt", 1))
    conversation_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                     f"{context['run_id']}:{context['dataset_scope']}:{stage}:{stage_attempt}"))
    context_path = context.get("preparation_context_path", "")
    stage_outputs = list(STAGE_OUTPUTS.get(stage, STAGE_OUTPUTS["candidate"]))
    source_manifest = context.get("source_manifests", {}).get(primary["id"])
    if not source_manifest:
        source_manifest = context.get("source_manifest")
    if not source_manifest:
        raise ValueError(f"Missing source manifest for source {primary['id']!r}")
    return DatasetAgentTask(
        task_id=f"{context['run_id']}:{context['dataset_scope']}:preparation", kind="preparation",
        source_root=str(Path(primary["root"]).resolve()),
        source_manifest=source_manifest,
        source_roots={item["id"]: item["root"] for item in sources},
        workspace=str(workspace), objective=objective,
         required_outputs=stage_outputs,
        feedback=feedback, attempt=stage_attempt, conversation_id=conversation_id,
        max_iterations=context.get("stage_max_iterations", context.get("coding_worker_max_iterations", 80)),
        timeout_seconds=context.get("stage_timeout_seconds", context.get("coding_worker_timeout_seconds", 3600)),
        model=get_settings().llm.model, base_url=get_settings().llm.base_url,
        reasoning_effort=get_settings().llm.reasoning_effort, run_id=context["run_id"],
        dataset_id=context["dataset_scope"], source_id=primary["id"], workflow_step=stage,
        stage=stage, context_path=context_path, allowed_outputs=stage_outputs,
        input_digest=context.get("preparation_input_digest", ""),
        progress_guard={"no_progress_limit": context.get("no_progress_limit", 8),
                        "tool_error_limit": context.get("tool_error_limit", 3)})

