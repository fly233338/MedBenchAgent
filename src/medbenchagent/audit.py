"""Deterministic post-generation benchmark audit; never judges medical truth."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from medbenchagent.generation import grounding, free_text

from medbenchagent.image_assets import validate_image_asset

from medbenchagent.answer_balance import balance_items
from medbenchagent.construction import AUDIT_AGENT, record_event
from medbenchagent.dataset import load_annotation_jsonl
from medbenchagent.generation.context import (
    locate_scalar_evidence,
    locate_bbox_evidence,
    scalar_evidence_candidates,
    has_meaningful_value,
    normalized_bbox,
    project_supporting_record,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()




def _image_dimensions(entry: dict[str, Any]) -> tuple[int, int] | None:
    width, height = entry.get("width"), entry.get("height")
    if isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0:
        return width, height
    return None


def _contract(
    item: dict[str, Any],
    task: dict[str, Any],
    dimensions: tuple[int, int] | None,
) -> list[str]:
    errors = []
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        errors.append("question")
    fmt = task.get("response_format")
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
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            or dimensions is not None
            and not (
                0 <= box[0] < box[2] <= dimensions[0]
                and 0 <= box[1] < box[3] <= dimensions[1]
            )
        ):
            errors.append("bbox")
    else:
        errors.append("response_format")
    return errors








def audit_run(
    *,
    raw_path: str | Path,
    blueprint_path: str | Path,
    events_path: str | Path,
    manifest_path: str | Path,
    annotation_path: str | Path,
    output_directory: str | Path,
    run_id: str,
    seed: int = 0,
) -> dict[str, str | int]:
    blueprint_bytes = Path(blueprint_path).read_bytes()
    blueprint = json.loads(blueprint_bytes)
    tasks = blueprint["tasks"]
    annotation_source = Path(annotation_path)
    annotation_rows = {
        record["image"]["image_id"]: record for record in load_annotation_jsonl(annotation_source)
    }
    manifest = {
        row["image_id"]: row
        for row in _rows(Path(manifest_path))
        if isinstance(row.get("image_id"), str)
    }
    image_errors: dict[str, list[str]] = {}
    candidates = []
    records = []
    for item in _rows(Path(raw_path)):
        item_id, task_id, image_id = item.get("id"), item.get("task_id"), item.get("image_id")
        errors: list[str] = []
        task = tasks.get(task_id) if isinstance(task_id, str) else None
        if item.get("dataset") != blueprint["dataset"]["dataset_id"] or not isinstance(task, dict):
            errors.append("item_task")
        entry = manifest.get(image_id)
        if entry is None:
            errors.append("image_manifest")
        else:
            image_errors.setdefault(image_id, validate_image_asset(entry, Path(manifest_path).parent))
            errors += image_errors[image_id]
        if isinstance(task, dict) and entry is not None:
            errors += _contract(item, task, _image_dimensions(entry))
        annotation_record = annotation_rows.get(image_id)
        if annotation_record is None:
            errors.append("annotation_source")
        elif isinstance(task, dict):
            fields = task.get("supporting_fields", [])
            try:
                if not fields or not all(has_meaningful_value(project_supporting_record(annotation_record, [field])) for field in fields):
                    errors.append("annotation_evidence")
                elif not _answer_matches(item, task, annotation_record):
                    errors.append("answer_evidence")
            except (ValueError, TypeError, KeyError):
                errors.append("answer_evidence")
        candidates.append((item, errors))
    seen_ids: set[object] = set()
    seen_items: set[str] = set()
    seen_questions: set[str] = set()
    duplicate_item_id_count = 0
    exact_item_duplicate_count = 0
    repeated_question_count = 0
    audited = []
    rejected = []
    for item, errors in candidates:
        item_id = item.get("id")
        canonical_item = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        question = item.get("question")
        normalized_question = _normal(question) if isinstance(question, str) else ""
        duplicate_id = item_id in seen_ids
        exact_duplicate = canonical_item in seen_items
        repeated_question = normalized_question in seen_questions
        duplicate_item_id_count += duplicate_id
        exact_item_duplicate_count += exact_duplicate
        repeated_question_count += repeated_question
        if not errors and (duplicate_id or exact_duplicate):
            errors.append("duplicate")
        if errors:
            rejected.append(item)
            status = "reject"
        else:
            audited.append(item)
            status = "accept"
        seen_ids.add(item_id)
        seen_items.add(canonical_item)
        seen_questions.add(normalized_question)
        records.append({"item_id": item.get("id"), "task_id": item.get("task_id"), "status": status, "reasons": errors})
        record_event(
            events_path,
            f"audit_{status}",
            run_id=run_id,
            dataset_id=item.get("dataset"),
            task_id=item.get("task_id"),
            item_id=item.get("id"),
            agent=AUDIT_AGENT,
            operation="deterministic",
            status=status,
        )
    output = Path(output_directory)
    prefix = f"benchmark.{blueprint['dataset']['dataset_id']}"
    balanced = balance_items(audited, seed)
    audited_path = output / f"{prefix}.audited.jsonl"
    report_path = output / "audit_report.json"
    _write(audited_path, balanced)
    report = {
        "schema_version": "audit-report.v2",
        "records": records,
        "input": len(candidates),
        "hard_audit": {
            "accepted": len(audited),
            "rejected": len(rejected),
            "reason_counts": dict(
                Counter(reason for record in records for reason in record["reasons"])
            ),
            "duplicate_statistics": {
                "duplicate_item_id_count": duplicate_item_id_count,
                "exact_item_duplicate_count": exact_item_duplicate_count,
                "repeated_question_count": repeated_question_count,
                "question_repetition_rate": (
                    repeated_question_count / len(candidates) if candidates else 0
                ),
            },
        },
        "semantic_audit": {
            "eligible": len(audited),
            "sampled": 0,
            "completed": 0,
            "passed": 0,
            "flagged": 0,
            "not_run": 0,
            "not_sampled": len(audited),
            "violation_counts": {},
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record_event(
        events_path,
        "audit_summary",
        run_id=run_id,
        dataset_id=blueprint["dataset"]["dataset_id"],
        agent=AUDIT_AGENT,
        operation="deterministic",
        status="complete",
        input=len(candidates),
        audited=len(audited),
        rejected=len(rejected),
        duplicate_statistics=report["hard_audit"]["duplicate_statistics"],
    )
    return {
        "audited_path": str(audited_path),
        "report_path": str(report_path),
        "item_count": len(audited),
    }


def _answer_matches(item: dict, task: dict, record: dict) -> bool:
    fields = task["supporting_fields"]
    fmt = task["response_format"]
    if fmt == "letter":
        values = scalar_evidence_candidates(record, task["answer_field"])
        if len(values) != 1:
            return False
        value = values[0]["value"]
        matches = [m["option"] for m in task["answer_value_mapping"]
                   if type(m["raw_value"]) is type(value) and m["raw_value"] == value]
        options, answer = item.get("options"), item.get("answer")
        return (len(matches) == 1 and options == task["options"]
                and isinstance(answer, str) and len(answer) == 1
                and 0 <= ord(answer) - 65 < len(options)
                and options[ord(answer) - 65] == matches[0])
    candidate = SimpleNamespace(record=record, task=task, task_id=item["task_id"],
                                supporting_record=project_supporting_record(record, fields))
    if fmt == "free_text":
        if free_text._reference_text(candidate) != item["answer"]:
            return False
        locate_scalar_evidence(record, fields, item["answer"])
        return True
    if fmt == "bbox":
        if not grounding.supports(candidate) or grounding._target_bbox(candidate) != normalized_bbox(item["answer"]):
            return False
        locate_bbox_evidence(record, fields, item["answer"])
        return normalized_bbox(item["answer"]) is not None
    return False
