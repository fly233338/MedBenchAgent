"""Adapter-code delegation after a semantic specification is frozen."""

from pathlib import Path

from medbenchagent.agents.coding_worker import CodingTask, OpenHandsCodingWorker
from medbenchagent.runtime.dependencies import publish

coding_worker = OpenHandsCodingWorker


def develop_dataset_adapter(context):
    workspace = Path(context["operation_directory"]) / "workers" / "adapter"
    task = CodingTask(task_id=f"{context.get('run_id', 'run')}:{context['source_scope']}:adapter",
                      kind="dataset_adapter", source_root=context["source_root"],
                      source_manifest=context["source_manifest"], workspace=str(workspace),
                      objective="Write and test adapter.py from the supplied semantic specification.",
                      required_outputs=["adapter.py"], model=context.get("model", ""),
                      base_url=context.get("base_url", ""), max_iterations=context.get("coding_worker_max_iterations", 40),
                      timeout_seconds=context.get("coding_worker_timeout_seconds", 1800),
                      conversation_id=context.get("adapter_worker_conversation_id") or None,
                      dataset_id=context.get("dataset_scope", ""), source_id=context["source_scope"])
    result = coding_worker(context).run(task)
    if result.status != "complete":
        return {"status": result.status, "summary": result.diagnostic or result.question, "updates": {}}
    adapter = result.outputs.get("adapter.py") or str(workspace / "adapter.py")
    reference = publish(context, "DatasetAdapter", {"path": adapter}, ("SemanticPreparationSpec",))
    return {"status": "success", "summary": "Dataset adapter candidate created",
            "updates": {"annotation_script_path": adapter, "adapter_worker_conversation_id": result.conversation_id,
                        "adapter_reference": (reference.get("artifact_versions", {}).get("DatasetAdapter", "")
                                              if isinstance(reference, dict) else ""),
                        "artifact_versions": reference.get("artifact_versions", {}) if isinstance(reference, dict) else {}}}
