"""Deterministic Generation validation and item quarantine for FinalizationAgent."""

import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from medbenchagent.data.image_assets import load_image_manifest, validate_image_asset
from medbenchagent.data.validation import load_annotation_jsonl
from medbenchagent.generation.templates import render, validate_templates
from medbenchagent.planning.protocol import Protocol, resolve_answer, template_specification
from medbenchagent.runtime.run_artifacts import atomic_write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_inputs(plan_path, templates_path, annotation_path, manifest_path):
    plan = read_json(plan_path)
    template_sets = read_json(templates_path)
    for leaf_id, task in plan["tasks"].items():
        Protocol.model_validate(task["protocol"])
        specification = template_specification(task["protocol"], plan["benchmark_info"]["language"])
        review = template_sets["reviews"][leaf_id]
        if review["specification"] != specification.model_dump() or review["review"]["verdict"] != "approved":
            raise ValueError("Templates are not approved for the current Plan")
        validate_templates(specification, template_sets["templates"][leaf_id])
    records = {}
    for record in load_annotation_jsonl(Path(annotation_path)):
        image_id = record["image"]["image_id"]
        if image_id in records:
            raise ValueError("Duplicate annotation image ID")
        records[image_id] = record
    return plan, template_sets["templates"], records, load_image_manifest(Path(manifest_path))


def item_errors(item, dataset_id, tasks, templates, annotations, images, image_root, image_errors, *, balanced=False):
    if not isinstance(item, dict):
        return ["item_object"]
    errors = []
    if not isinstance(item.get("id"), str) or not item["id"].strip():
        errors.append("item_id")
    task_id, image_id = item.get("task_id"), item.get("image_id")
    task = tasks.get(task_id) if isinstance(task_id, str) else None
    entry = images.get(image_id) if isinstance(image_id, str) else None
    annotation = annotations.get(image_id) if isinstance(image_id, str) else None
    if item.get("dataset") != dataset_id or task is None:
        errors.append("item_task")
    if entry is None:
        errors.append("image_manifest")
    else:
        if image_id not in image_errors:
            image_errors[image_id] = validate_image_asset({"png_path": entry.relative_png_path,
                "width": entry.width, "height": entry.height}, image_root)
        errors.extend(image_errors[image_id])
    if annotation is None:
        errors.append("annotation_source")
    if task is not None:
        errors.extend(_contract(item, task, (entry.width, entry.height) if entry else None))
        if annotation is not None:
            try:
                if not _answer_matches(item, task, annotation, balanced=balanced):
                    errors.append("answer_evidence")
                rendered = {render(SimpleNamespace(record=annotation, task=task, question_template=t)).strip()
                            for t in templates[task_id]}
                if item.get("question") not in rendered:
                    errors.append("template_rendering")
            except (ValueError, TypeError, KeyError):
                errors.append("answer_evidence")
    return sorted(set(errors))


def inspect_generation(*, dataset_id, raw_path, plan_path, templates_path, annotation_path, manifest_path, output_directory):
    plan, templates, annotations, images = load_inputs(plan_path, templates_path, annotation_path, manifest_path)
    if plan["dataset"]["dataset_id"] != dataset_id:
        raise ValueError("Plan dataset identity differs from Generation")
    accepted, rejected, image_errors = [], [], {}
    counts = {key: {"generated": 0, "accepted": 0, "rejected": 0} for key in plan["tasks"]}
    unassigned = {"generated": 0, "accepted": 0, "rejected": 0}
    reasons = Counter()
    with Path(raw_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = {"dataset_id": dataset_id, "path": str(raw_path), "line": line_number, "raw_text": line.rstrip("\r\n")}
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item, errors = None, ["invalid_json"]
            else:
                errors = item_errors(item, dataset_id, plan["tasks"], templates, annotations, images,
                                     Path(manifest_path).parent, image_errors)
            leaf_id = item.get("task_id") if isinstance(item, dict) else None
            bucket = counts[leaf_id] if isinstance(leaf_id, str) and leaf_id in counts else unassigned
            bucket["generated"] += 1
            if errors:
                bucket["rejected"] += 1
                reasons.update(errors)
                rejected.append({"source": source, "task_id": leaf_id if isinstance(leaf_id, str) else None, "reasons": errors})
            else:
                bucket["accepted"] += 1
                accepted.append({"source": source, "item": item})
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "accepted.jsonl", accepted)
    write_rows(root / "rejected.jsonl", rejected)
    report = {"generated": len(accepted) + len(rejected), "accepted": len(accepted), "rejected": len(rejected),
              "tasks": counts, "unassigned": unassigned, "reason_counts": dict(reasons)}
    atomic_write_json(root / "report.json", report)
    return {"accepted_path": str(root / "accepted.jsonl"), "rejected_path": str(root / "rejected.jsonl"),
            "report_path": str(root / "report.json")}


def _contract(
    item: dict[str, Any],
    task: dict[str, Any],
    dimensions: tuple[int, int] | None,
) -> list[str]:
    errors = []
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        errors.append("question")
    fmt = task["protocol"]["response_format"]
    if fmt == "letter":
        options, answer = item.get("options"), item.get("answer")
        if (
            not isinstance(options, list)
            or not all(isinstance(option, str) and option.strip() for option in options)
            or len(set(options)) != len(options)
            or not isinstance(answer, str)
            or len(answer) != 1
            or not answer.isalpha()
            or ord(answer.upper()) - 65 not in range(len(options))
        ):
            errors.append("letter_answer")
    elif fmt == "free_text":
        if not isinstance(item.get("answer"), str) or not item["answer"].strip():
            errors.append("free_text_answer")
    elif fmt == "bbox":
        box = item.get("answer")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(v) in (int, float) and math.isfinite(v) for v in box)
            or dimensions is not None
            and not (
                0 <= box[0] < box[2] <= dimensions[0]
                and 0 <= box[1] < box[3] <= dimensions[1]
            )
        ):
            errors.append("bbox")
    else:
        errors.append("response_format")
    if fmt != "letter" and item.get("options") is not None:
        errors.append("unexpected_options")
    return errors

def _answer_matches(item: dict, task: dict, record: dict, *, balanced=False) -> bool:
    resolved = resolve_answer(record, task["protocol"])
    if resolved is None:
        return False
    if task["protocol"]["response_format"] == "letter":
        options, answer = item.get("options"), item.get("answer")
        return (isinstance(options, list) and (set(options) == set(task["protocol"]["answer_contract"]["options"]) if balanced else options == task["protocol"]["answer_contract"]["options"])
                and isinstance(answer, str) and len(answer) == 1
                and 0 <= ord(answer) - 65 < len(options)
                and options[ord(answer) - 65] == resolved["answer"])
    return item.get("answer") == resolved["answer"]
