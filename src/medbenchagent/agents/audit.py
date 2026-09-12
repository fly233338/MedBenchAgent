"""AuditAgent tools retain deterministic checks and sampled semantic screening."""

import json
from pathlib import Path
from medbenchagent.audit.hard import audit_run
from medbenchagent.runtime.construction import event_context
from medbenchagent.runtime.harness import result
from medbenchagent.audit.semantic_audit import semantic_audit_run
from medbenchagent.runtime.dependencies import publish, require


def run_audit(context):
    """Audit the current generated batch and return actionable problems."""
    require(context, "Generation")
    folder = Path(context["operation_directory"]) / "audit"
    folder.mkdir(parents=True, exist_ok=True)
    report = audit_run(
        raw_path=context["raw_output_path"],
        blueprint_path=context["working_blueprint_path"],
        events_path=context["construction_events_path"],
        manifest_path=Path(context["annotation_path"]).parent / "image_manifest.jsonl",
        annotation_path=context["annotation_path"],
        output_directory=folder,
        run_id=context["run_id"],
        seed=context.get("seed", 0),
    )
    with event_context(agent="AuditAgent"):
        semantic_audit_run(
            audited_path=report["audited_path"],
            report_path=report["report_path"],
            blueprint_path=context["working_blueprint_path"],
            events_path=context["construction_events_path"],
            output_directory=folder,
            run_id=context["run_id"],
            dataset_id=context["selected_dataset_id"],
            seed=context.get("seed", 0),
        )
    payload = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    return result(
        "success",
        "Audit complete; inspect rejected counts and semantic flags",
        **publish(
            context, "Audit", {"audited": report["audited_path"], "report": report["report_path"]}
        ),
        audit_artifacts={"audited": report["audited_path"], "report": report["report_path"]},
        audit_summary={
            "hard_audit": payload["hard_audit"],
            "semantic_audit": payload["semantic_audit"],
            "issues": [r for r in payload["records"] if r["reasons"]][:20],
        },
    )
