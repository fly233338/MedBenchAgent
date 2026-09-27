"""One coding task replaces the raw source's exploratory inspect loop."""

from __future__ import annotations

import shutil
from pathlib import Path

from medbenchagent.agents.coding_worker import CodingTask, coding_worker
from medbenchagent.agents.preparation import save_source_fingerprint
from medbenchagent.data.source_report import (JsonPopulationBinding, SourceReport,
                                              SpecialPopulationBinding, validate_source_report)
from medbenchagent.runtime.dependencies import require
from medbenchagent.runtime.harness import result
from medbenchagent.runtime.inspection import save_evidence
from medbenchagent.runtime.run_artifacts import atomic_write_json
from medbenchagent.settings import get_settings


def build_source_inventory(context):
    """Ask OpenHands for one complete inventory, then validate it before advancing."""
    source = context["active_source"]
    if source["kind"] != "raw":
        raise ValueError("Coding inventory is only for raw sources")
    require(context, "Source")
    source_version = next(iter(require(context, "Source").values()))["id"]
    workspace = (Path(context["run_directory"]) / "workers" / "openhands" /
                 context["dataset_scope"] / source["id"])
    workspace.mkdir(parents=True, exist_ok=True)
    schema_path = Path(context["operation_directory"]) / "source_report.schema.json"
    atomic_write_json(schema_path, SourceReport.model_json_schema())
    attempts = context.get("source_inventory_attempts", 0) + 1
    feedback = context.get("source_inventory_feedback", "")
    clarification = context.get("clarification") or {}
    if context.get("source_inventory_question") and clarification.get("status") == "resolved":
        from medbenchagent.runtime.clarifications import answer_payload

        feedback += "\nUser clarification: " + answer_payload(context, clarification)["text"]
    objective = (
        f"Inspect raw source {source['id']} completely. Produce source_report.json according to "
        "inputs/source_report.schema.json, image_population.jsonl, and inspect_source.py. "
        "For a tabular population, every source row must yield exactly one JSONL line with keys "
        "source_id, row_id, image_id, image_path, mask_paths, classification. "
        "For a direct JSON/JSONL object collection, use the json population kind and the same "
        "six JSONL keys for every selected object. record_count is the full object or table row count. "
        "inspect_source.py must read MEDBENCH_SOURCE_ROOT and MEDBENCH_WORK_DIR, then write "
        "source_report.json and image_population.jsonl under MEDBENCH_WORK_DIR. "
        "Use the exact raw classification strings, including empty strings, without interpreting them. "
        "Use source-relative paths, confirm every image and nonempty mask exists, and include all rows. "
        "If a table cannot independently enumerate the intended image population, use the special "
        "population schema and provide one image per JSONL line with source_id, image_id, image_path, "
        "mask_paths. Mark the actual fixed inspector capability gap; the candidate list will require "
        "MedBench human review and must not claim proven completeness."
    )
    task = CodingTask(
        task_id=f"{context['run_id']}:{context['dataset_scope']}:{source['id']}:{source_version}:inventory",
        kind="source_inventory", source_root=context["source_root"],
        source_manifest=context["source_manifest"], workspace=str(workspace),
        objective=objective, required_outputs=["source_report.json", "image_population.jsonl", "inspect_source.py"],
        inputs={"source_report.schema.json": str(schema_path)},
        feedback=feedback,
        attempt=attempts,
        conversation_id=(context.get("source_inventory_conversation_id")
                         if context.get("source_inventory_question") and clarification.get("status") == "resolved"
                         else None),
        max_iterations=context.get("coding_worker_max_iterations", 40),
        timeout_seconds=context.get("coding_worker_timeout_seconds", 1800),
        model=get_settings().llm.model, base_url=get_settings().llm.base_url,
        reasoning_effort=get_settings().llm.reasoning_effort,
        construction_events_path=context["construction_events_path"], run_id=context["run_id"],
        dataset_id=context["dataset_scope"], source_id=source["id"],
    )
    worker_result = coding_worker(context).run(task)
    updates = {"source_inventory_conversation_id": worker_result.conversation_id,
               "source_inventory_attempts": attempts}
    if worker_result.status != "complete":
        diagnostic = worker_result.question or worker_result.diagnostic or "Coding worker did not finish inventory"
        if worker_result.question:
            updates["source_inventory_question"] = diagnostic
        status = ("blocked" if worker_result.status == "blocked"
                  or attempts >= context.get("coding_worker_max_revisions", 3) else "failed")
        return result(status, diagnostic, source_inventory_feedback=diagnostic,
                      source_inventory_error_category=worker_result.error_category,
                      source_inventory_retryable=worker_result.retryable, **updates)
    try:
        verified = validate_source_report(context, worker_result.outputs["source_report.json"],
                                          worker_result.outputs["image_population.jsonl"],
                                          Path(context["operation_directory"]) / "verified")
        script = Path(worker_result.outputs["inspect_source.py"]).read_text(encoding="utf-8")
        compile(script, "inspect_source.py", "exec")
    except (ValueError, SyntaxError) as exc:
        feedback = str(exc)
        return result("blocked" if attempts >= context.get("coding_worker_max_revisions", 3) else "failed",
                      f"Source report validation failed: {feedback}",
                      source_inventory_feedback=feedback,
                      source_inventory_error_category="worker_output_invalid",
                      source_inventory_retryable=True, **updates)
    report = verified["report"]
    stable_report = Path(context["operation_directory"]) / "source_report.json"
    stable_script = Path(context["operation_directory"]) / "inspect_source.py"
    shutil.copyfile(worker_result.outputs["source_report.json"], stable_report)
    shutil.copyfile(worker_result.outputs["inspect_source.py"], stable_script)
    public = {"tables": verified["table_summaries"],
              "population": report.population.model_dump(), "rows": verified["rows"],
              "unique_images": verified["unique_images"],
              "mask_references": verified["mask_references"],
              "classification_counts": verified["classification_counts"],
              "columns": verified["columns"], "samples": verified["samples"],
              "uncertainties": report.uncertainties, "assurance": verified["assurance"]}
    special = isinstance(report.population, SpecialPopulationBinding)
    structure_operation = ("source_structure" if special else
                           "json_structure" if isinstance(report.population, JsonPopulationBinding) else "table")
    table_ref = save_evidence(context, structure_operation,
                              report.population.model_dump(), public, ("Source",), source_ids=[source["id"]])
    population_value = {"unique_images": verified["unique_images"], "population_path": verified["population_path"],
                        "population_digest": verified["population_digest"],
                        "coverage": "candidate_only" if special else "full",
                        "rows": verified["rows"], "mask_references": verified["mask_references"]}
    if special:
        population_value.update(supported=False,
                                required_operation=report.population.required_operation,
                                reason=report.population.reason,
                                inspection_reference=table_ref)
    population_ref = save_evidence(context, "population_capability" if special else "image_population",
                                   report.population.model_dump(), population_value,
                                   ("Source",), source_ids=[source["id"]])
    context["source_report_path"] = str(stable_report)
    frozen = save_source_fingerprint(context, inspection_references=[table_ref, population_ref])
    updates.update(frozen["updates"])
    updates.update(evidence_index_path=str(Path(context["run_directory"]) / "evidence" / "index.sqlite"),
                   source_report_path=str(stable_report),
                   source_report_reference=table_ref, source_population_reference=population_ref,
                   source_inspection_script_path=str(stable_script),
                   source_inventory_feedback="", source_inventory_question="")
    return result("success", f"Verified {verified['rows']} rows, {verified['unique_images']} images and "
                  f"{verified['mask_references']} masks; source facts frozen", **updates)
