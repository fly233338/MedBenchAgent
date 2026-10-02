"""Run-context adapter for deterministic v3 Task Plan execution."""

from __future__ import annotations

import json
from pathlib import Path

from medbenchagent.planning.pipeline_v2 import run_v2_pipeline
from medbenchagent.planning.task_space import TaskPlanState
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.planning.bir import BIRStore, digest_file


def run_task_space_pipeline(context, *, requested_count: int | None = None):
    access = context.get("dataset_access_manifest", {})
    if access.get("schema_version") != "annotation.v5":
        raise ValueError("Task Space pipeline requires annotation.v5")
    operation = Path(context["operation_directory"])
    plan_path = context.get("task_plan_path") or context.get("task_space_plan_path")
    if not plan_path or not Path(plan_path).is_file():
        raise ValueError("Axis Analysis/Task Plan artifact is required")
    plan = TaskPlanState.model_validate(json.loads(Path(plan_path).read_text(encoding="utf-8")))
    if plan.phase != "complete":
        raise ValueError("Task Plan must be complete before execution")
    from medbenchagent.data.v5_index import V5DatasetQuery
    cases = list(V5DatasetQuery(access).iter_cases())
    output = run_v2_pipeline(
        dataset={"dataset_id": context.get("selected_dataset_id", "dataset")},
        plan=plan,
        cases=cases,
        requested_count=requested_count or context.get("benchmark_info", {}).get("sample_count", 1),
        seed=context.get("seed", 0),
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
    if context.get("bir_path"):
        store = BIRStore(context["bir_path"])
        dataset_id = context.get("selected_dataset_id", "dataset")
        manifest = json.loads((store.root / "manifest.json").read_text(encoding="utf-8"))
        allocation_value = {
            "schema_version": "bir-allocation.v1",
            "status": "complete",
            "revision": int(manifest["revision"]) + 1,
            "requested_count": requested_count or context.get("benchmark_info", {}).get("sample_count", 1),
            "assignments": {dataset_id: output["allocation"]},
            "counts": output["allocation"],
            "shortfall": output.get("shortfall", {}),
            "dependencies": {"planning": digest_file(plan_path)},
        }
        output_value = {
            "schema_version": "bir-output.v1",
            "dataset_id": dataset_id,
            "status": "complete",
            "benchmark_items_path": paths["benchmark_items.json"],
            "benchmark_items_digest": digest_file(paths["benchmark_items.json"]),
            "audit_path": paths["task_space_audit.json"],
            "manifest_path": paths["task_space_manifest.json"],
            "capability_report_path": paths["capability_report.json"],
            "counts": output["audit"],
            "shortfall": output.get("shortfall", {}),
            "dependencies": {"planning": digest_file(plan_path)},
        }
        delivery_value = {
            "schema_version": "bir-delivery.v1",
            "status": "complete",
            "outputs": {dataset_id: f"outputs/{dataset_id}.json"},
            "counts": {"accepted": output["audit"].get("accepted", 0), "generated": len(output["items"])},
            "validation": output["audit"],
            "dependencies": {"allocation": "allocation.json", "output": f"outputs/{dataset_id}.json"},
        }
        commit = store.transaction({"allocation.json": allocation_value,
                                    f"outputs/{dataset_id}.json": output_value,
                                    "delivery.json": delivery_value},
                                   run_id=context.get("run_id", "unknown"), actor="v2_pipeline",
                                   expected_revision=int(context.get("bir_revision", manifest["revision"])))
        context.update(bir_revision=commit.revision, bir_manifest_digest=commit.manifest_digest)
    return result(
        "success",
        f"Task Space pipeline completed with {len(output['items'])} items",
        **paths,
        task_space_output=output,
        bir_revision=context.get("bir_revision"), bir_manifest_digest=context.get("bir_manifest_digest"),
        bir_summary_ref=({"revision": context.get("bir_revision"), "dataset_id": context.get("selected_dataset_id", "dataset"),
                          "delivery": "delivery.json"} if context.get("bir_path") else None),
    )

