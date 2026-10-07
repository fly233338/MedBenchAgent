"""Inspect canonical schemas or run the same read-only checks as Host."""
import argparse
import json
from pathlib import Path
from .contracts import Stage, PreparationMapping, ImageConversionPlan, Submission
from .validation import validate_stage
from medbenchagent.data.models import CaseRecord, ImageEntry


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema")
    schema.add_argument("name", choices=["mapping", "conversion", "case", "image", "submission"])
    check = sub.add_parser("validate")
    check.add_argument("--request", default="request.json")
    check.add_argument("--candidate", default="work/outputs")
    args = parser.parse_args(argv)
    if args.command == "schema":
        model = {"mapping": PreparationMapping, "conversion": ImageConversionPlan, "case": CaseRecord,
                 "image": ImageEntry, "submission": Submission}[args.name]
        print(json.dumps(model.model_json_schema(), ensure_ascii=False))
        return 0
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
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
    return 0 if result.status == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
