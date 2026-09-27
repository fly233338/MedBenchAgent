"""Delegate generated extractor development without transferring publication authority."""

from __future__ import annotations

from pathlib import Path

from medbenchagent.agents.coding_worker import CodingTask, coding_worker
from medbenchagent.agents.preparation import save_dataset_adapter
from medbenchagent.data.contracts import ExtractionProvenance
from medbenchagent.data.preparation import CAPABILITIES
from medbenchagent.runtime.dependencies import require
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


ADAPTER_CONTRACT = """Write adapter.py for the current SemanticPreparationSpec.
At runtime read MEDBENCH_SOURCE_ROOT, MEDBENCH_WORK_DIR, MEDBENCH_SPEC_PATH and MEDBENCH_INPUTS_PATH.
Write annotation.jsonl containing one JSON object per selected image: {"record": canonical_annotation,
"record_ref": unique_string}. Write provenance.jsonl with one extraction-provenance.v1 object per record.
Each provenance object needs id equal to record_ref, one or more real source-locator.v1 locators,
findings locators for finding objects, and a step identifier. Do not assign trusted provenance,
namespaces, quality decisions, publication flags or output image transformations.
Use only source files and declared semantic decisions. Preserve missing versus negative.
Run the script against the read-only source and inspect its output before finishing.
"""


def develop_dataset_adapter(context, *, feedback=""):
    """Use OpenHands to write and repair a candidate Python adapter for the current spec."""
    require(context, "SemanticPreparationSpec")
    source = context["active_source"]
    spec_version = next(iter(require(context, "SemanticPreparationSpec").values()))["id"]
    attempts = (context.get("adapter_worker_attempts", 0)
                if context.get("adapter_worker_spec_version") == spec_version else 0) + 1
    if attempts > context.get("coding_worker_max_revisions", 3):
        return result("blocked", "Adapter worker revision budget exhausted")
    workspace = (Path(context["run_directory"]) / ".runtime" / "workers" / "openhands" /
                 context["dataset_scope"] / source["id"])
    contract = Path(context["operation_directory"]) / "adapter_contract.txt"
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(ADAPTER_CONTRACT, encoding="utf-8")
    inputs_path = Path(context["operation_directory"]) / "adapter_inputs.json"
    atomic_write_json(inputs_path, {"source": source, "capabilities": CAPABILITIES,
                                    "provenance_schema": ExtractionProvenance.model_json_schema()})
    task = CodingTask(
        task_id=f"{context['run_id']}:{context['dataset_scope']}:{source['id']}:{spec_version}:adapter",
        kind="dataset_adapter", source_root=context["source_root"],
        source_manifest=context["source_manifest"], workspace=str(workspace),
        objective="Implement the current specification as outputs/adapter.py. Read inputs/adapter_contract.txt "
                  "and inputs/spec.json. Return code only under outputs/; MedBench will validate and publish it.",
        required_outputs=["adapter.py"],
        inputs={"adapter_contract.txt": str(contract), "spec.json": context["semantic_spec_path"],
                "adapter_inputs.json": str(inputs_path)},
        feedback=feedback,
        attempt=attempts,
        conversation_id=(context.get("adapter_worker_conversation_id")
                         if not feedback and context.get("adapter_worker_spec_version") == spec_version else None),
        max_iterations=context.get("coding_worker_max_iterations", 40),
        timeout_seconds=context.get("coding_worker_timeout_seconds", 1800),
        model=get_settings().llm.model, base_url=get_settings().llm.base_url,
        construction_events_path=context["construction_events_path"], run_id=context["run_id"],
        dataset_id=context["dataset_scope"], source_id=source["id"],
    )
    outcome = coding_worker(context).run(task)
    updates = {"adapter_worker_conversation_id": outcome.conversation_id,
               "adapter_worker_spec_version": spec_version,
               "adapter_worker_attempts": attempts}
    if outcome.status != "complete":
        return result("blocked" if outcome.status == "blocked" or attempts >= context.get("coding_worker_max_revisions", 3)
                      else "failed",
                      outcome.question or outcome.diagnostic or "Adapter worker did not finish", **updates)
    code = Path(outcome.outputs["adapter.py"]).read_text(encoding="utf-8")
    try:
        saved = save_dataset_adapter(context, code=code)
    except ValueError as exc:
        return result("blocked" if attempts >= context.get("coding_worker_max_revisions", 3) else "failed",
                      f"Adapter validation failed: {exc}", **updates)
    return result("success", "OpenHands adapter saved for MedBench validation",
                  **updates, **saved["updates"])
