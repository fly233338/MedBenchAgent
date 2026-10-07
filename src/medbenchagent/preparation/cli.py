"""Inspect schemas, check candidates, and report exceptional stage control."""
import argparse
import json
from pathlib import Path
from .contracts import Stage, PreparationMapping, ImageConversionPlan, StageControl
from .validation import validate_stage
from medbenchagent.data.models import CaseRecord, ImageEntry
from medbenchagent.runtime.run_artifacts import atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema")
    schema.add_argument("name", choices=["mapping", "conversion", "case", "image", "control", "question"])
    for command in ("validate", "signal"):
        check = sub.add_parser(command)
        check.add_argument("--request", default="request.json")
        check.add_argument("--candidate", default="work/outputs")
        if command == "signal":
            check.add_argument("--outcome", choices=["needs_input", "blocked"], required=True)
            check.add_argument("--summary", default="")
            check.add_argument("--questions-file", help="JSON array of SemanticQuestion records")
            check.add_argument("--diagnostic-code", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "schema":
        from .contracts import SemanticQuestion
        model = {"mapping": PreparationMapping, "conversion": ImageConversionPlan, "case": CaseRecord,
                 "image": ImageEntry, "control": StageControl, "question": SemanticQuestion}[args.name]
        print(json.dumps(model.model_json_schema(), ensure_ascii=False))
        return 0
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    control_path = Path(args.request).resolve().parent / "work/control.json"
    if args.command == "signal":
        # Replace only exceptional control; there is no success receipt.
        control_path.unlink(missing_ok=True)
        questions = json.loads(Path(args.questions_file).read_text(encoding="utf-8")) if args.questions_file else []
        signal = StageControl.model_validate({
            "requested_outcome": args.outcome, "summary": args.summary,
            "questions": questions, "diagnostic_codes": args.diagnostic_code,
        }, strict=False)
        if args.outcome == "needs_input" and not signal.questions:
            parser.error("needs_input requires --questions-file with scoped questions")
        if any(question.stage.value != request["stage"] for question in signal.questions):
            parser.error("Questions must belong to the current stage")
        atomic_write_json(control_path, signal.model_dump(mode="json"))
        print(signal.model_dump_json(indent=2))
        return 0
    context = request["context"]
    # Paths in context are worker-relative, including in Docker.
    refs = {key: {"name": key, "path": value, "digest": __import__("hashlib").sha256(Path(value).read_bytes()).hexdigest()} for key, value in context["inputs"].items()}
    state = {key: context[key] for key in ("source_roots", "source_manifests", "locator_roots", "bundle_root")}
    if refs:
        state["accepted_handoff"] = {"dataset_id": context["dataset_id"], "source_snapshot_digest": request["source_snapshot_digest"],
            "mapping": refs["mapping.json"], "image_conversion_plan": refs["image_conversion_plan.json"], "digest": request["input_digest"]}
    if request["stage"] == "prepare_image":
        state["text_root"] = str(Path(context["inputs"]["cases.jsonl"]).parent)
        state["text_artifacts"] = {name: ref for name, ref in refs.items() if name in {"cases.jsonl", "metadata.json"}}
    result = validate_stage(args.candidate, Stage(request["stage"]), state)
    print(result.model_dump_json(indent=2))
    print("Candidate validation only. Host validates outputs after the worker ends.")
    return 0 if result.status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
