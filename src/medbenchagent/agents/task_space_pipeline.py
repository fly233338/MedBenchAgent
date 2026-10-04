"""Run-context adapter retained for in-process callers.

The Supervisor no longer uses this one-shot helper: global Allocation and
Finalization are explicit stages.  This function only materializes a local
pipeline result for focused callers and never writes global BIR parts.
"""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.planning.pipeline import run_pipeline
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json


def run_task_space_pipeline(context, *, requested_count: int | None = None):
    access = context.get("dataset_access_manifest", {})
    if access.get("schema_version") != "annotation":
        raise ValueError("Task Space pipeline requires annotation")
    operation = Path(context["operation_directory"])
    plan_path = context.get("task_plan_path") or context.get("task_space_plan_path")
    if not plan_path or not Path(plan_path).is_file():
        raise ValueError("Axis Analysis/Task Plan artifact is required")
    plan = TaskPlanState.model_validate(json.loads(Path(plan_path).read_text(encoding="utf-8")))
    if plan.phase != "complete":
        raise ValueError("Task Plan must be complete before execution")
    from medbenchagent.data.index import DatasetIndex
    from medbenchagent.planning.resolver import ResolverContext
    cases = list(DatasetIndex(access).iter_cases())
    output = run_pipeline(
        dataset={"dataset_id": context.get("selected_dataset_id", "dataset")},
        plan=plan,
        cases=cases,
        requested_count=requested_count or context.get("benchmark_info", {}).get("sample_count", 1),
        seed=context.get("seed", 0),
        resolver_context=ResolverContext.from_access(access),
    )
    paths = {}
    for name, payload in (
        ("benchmark_items.json", output["items"]),
        ("task_space_audit.json", output["audit"]),
        ("task_space_manifest.json", output["manifest"]),
        ("capability_report.json", output["capability_report"]),
    ):
        path = operation / name
        atomic_write_json(path, payload)
        paths[name] = str(path)
    return result(
        "success",
        f"Task Space pipeline completed with {len(output['items'])} items",
        **paths,
        task_space_output=output,
        bir_revision=context.get("bir_revision"), bir_manifest_digest=context.get("bir_manifest_digest"),
    )

