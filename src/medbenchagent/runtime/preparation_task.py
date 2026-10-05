"""Build the single worker request for a preparation stage."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from medbenchagent.preparation.contracts import STAGE_SPECS, Stage
from medbenchagent.settings import get_settings


STAGE_OUTPUTS = {stage: list(spec.required_outputs) for stage, spec in STAGE_SPECS.items()}


def build_data_preparation_task(
    context: dict[str, Any],
    workspace: Path,
    *,
    feedback: str = "",
    stage: str = "prepare",
    attempt: int | None = None,
    execution_key: str = "",
):
    """Create a DatasetAgentTask without adding hidden prompt text."""
    from medbenchagent.runtime.dataset_runtime import DatasetAgentTask

    current = Stage(stage)
    sources = context.get("sources", [])
    if not sources:
        raise ValueError("Dataset preparation requires at least one source")
    primary = context.get("active_source") or sources[0]
    primary_id = primary.get("id") if isinstance(primary, dict) else getattr(primary, "id", "")
    source_manifests = context.get("source_manifests", {})
    source_manifest = source_manifests.get(primary_id) if isinstance(source_manifests, dict) else None
    if not source_manifest:
        source_manifest = context.get("source_manifest")
    if not source_manifest:
        raise ValueError(f"Missing source manifest for source {primary_id!r}")
    stage_attempt = attempt or int(context.get("preparation_attempt", 1))
    run_id = str(context.get("run_id") or "run")
    dataset_id = str(context.get("dataset_scope") or context.get("dataset_id") or "dataset")
    conversation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{dataset_id}:{current.value}:{stage_attempt}"))
    stage_dir = Path(workspace)
    prompt_path = context.get("preparation_prompt_path") or str(stage_dir / "inputs" / "prompt.md")
    context_path = context.get("preparation_context_path") or str(stage_dir / "inputs" / "context.json")
    settings = get_settings()
    data_settings = settings.data_processing
    llm = settings.llm
    budget = context.get("preparation_budget") or {}
    source_roots = context.get("source_roots") or {
        item.get("id"): item.get("root") for item in sources if isinstance(item, dict)
    }
    primary_root = primary.get("root") if isinstance(primary, dict) else primary.root
    return DatasetAgentTask(
        task_id=f"{run_id}:{dataset_id}:{current.value}",
        kind="preparation",
        source_root=str(Path(primary_root).resolve()),
        source_manifest=str(source_manifest),
        source_roots={str(key): str(value) for key, value in source_roots.items()},
        workspace=str(stage_dir),
        objective=STAGE_SPECS[current.value].objective,
        required_outputs=list(STAGE_SPECS[current.value].required_outputs),
        feedback=feedback,
        attempt=stage_attempt,
        conversation_id=conversation_id,
        max_iterations=int(budget.get("max_iterations", context.get("stage_max_iterations", data_settings.get("coding_worker_max_iterations", 80)))),
        timeout_seconds=int(budget.get("timeout_seconds", context.get("stage_timeout_seconds", data_settings.get("coding_worker_timeout_seconds", 3600)))),
        model=llm.model,
        base_url=llm.base_url,
        reasoning_effort=llm.reasoning_effort,
        run_id=run_id,
        dataset_id=dataset_id,
        source_id=str(primary_id),
        workflow_step=current.value,
        stage=current.value,
        context_path=str(context_path),
        prompt_path=str(prompt_path),
        allowed_outputs=list(STAGE_SPECS[current.value].required_outputs + STAGE_SPECS[current.value].optional_outputs),
        input_digest=str(context.get("preparation_input_digest") or context.get("input_digest") or "unknown-input"),
        execution_key=execution_key or str(context.get("preparation_execution_key") or "unkeyed"),
        construction_events_path=str(context.get("construction_events_path") or ""),
        resume_existing=bool(context.get("resume_existing", False)),
        progress_guard={"no_progress_limit": int(context.get("no_progress_limit", data_settings.get("preparation_no_progress_limit", 8))),
                        "tool_error_limit": int(context.get("tool_error_limit", data_settings.get("preparation_tool_error_limit", 3)))},
    )
