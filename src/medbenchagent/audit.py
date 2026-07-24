"""Deterministic post-generation benchmark audit; never judges medical truth."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from medbenchagent.answer_balance import balance_items
from medbenchagent.construction import record_event
from medbenchagent.run_artifacts import blueprint_sha256


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _image_errors(entry: dict[str, Any], root: Path) -> list[str]:
    png = entry.get("png_path")
    if not isinstance(png, str):
        return ["image_manifest_path"]
    path = PurePosixPath(png)
    if path.is_absolute() or ".." in path.parts:
        return ["image_manifest_path"]
    image_path = root.joinpath(*path.parts)
    if not image_path.is_file() or image_path.stat().st_size == 0:
        return ["image_missing"]
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != entry.get("sha256"):
        return ["image_hash"]
    try:
        with Image.open(image_path) as image:
            image.load()
            if image.mode != "L" or image.width <= 0 or image.height <= 0:
                return ["image_format"]
            if image.width != entry.get("width") or image.height != entry.get("height"):
                return ["image_dimensions"]
            values = list(image.getdata())
            if not values or len(set(values)) == 1:
                return ["image_pixels"]
    except (OSError, ValueError):
        return ["image_decode"]
    return []


def _contract(item: dict[str, Any], task: dict[str, Any], dimensions: tuple[int, int]) -> list[str]:
    errors = []
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        errors.append("question")
    fmt = task.get("response_format")
    if fmt == "letter":
        options, answer = item.get("options"), item.get("answer")
        if (
            not isinstance(options, list)
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
            or not (0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1])
        ):
            errors.append("bbox")
    else:
        errors.append("response_format")
    return errors


def audit_run(
    *,
    raw_path: str | Path,
    blueprint_path: str | Path,
    provenance_path: str | Path,
    events_path: str | Path,
    manifest_path: str | Path,
    output_directory: str | Path,
    run_id: str,
    seed: int = 0,
) -> dict[str, str | int]:
    blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
    tasks = blueprint["tasks"]
    manifest = {
        row["image_id"]: row
        for row in _rows(Path(manifest_path))
        if isinstance(row.get("image_id"), str)
    }
    provenance = {row.get("item_id"): row for row in _rows(Path(provenance_path))}
    call_ids = {
        row.get("call_id")
        for row in _rows(Path(events_path))
        if row.get("event_type") == "llm_call"
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
            image_errors.setdefault(image_id, _image_errors(entry, Path(manifest_path).parent))
            errors += image_errors[image_id]
        if isinstance(task, dict) and entry is not None:
            errors += _contract(item, task, (entry.get("width", 0), entry.get("height", 0)))
        source = provenance.get(item_id)
        if (
            not isinstance(source, dict)
            or source.get("image_id") != image_id
            or source.get("task_id") != task_id
            or source.get("dataset_id") != item.get("dataset")
            or (
                isinstance(task, dict)
                and source.get("supporting_fields") != task.get("supporting_fields")
            )
            or source.get("blueprint_hash") != blueprint_sha256(blueprint_path)
        ):
            errors.append("provenance")
        elif (
            not set(source.get("generation_call_ids", []) + source.get("review_call_ids", []))
            <= call_ids
        ):
            errors.append("provenance_calls")
        candidates.append((item, errors))
    seen: set[tuple[object, ...]] = set()
    audited = []
    rejected = []
    for item, errors in candidates:
        key_set = [
            ("id", item.get("id")),
            ("question", item.get("question")),
            ("normalized_question", _normal(item.get("question", ""))),
            ("task_image", item.get("dataset"), item.get("task_id"), item.get("image_id")),
            (
                "image_question_answer",
                item.get("image_id"),
                item.get("question"),
                json.dumps(item.get("answer"), sort_keys=True),
            ),
        ]
        if not errors and any(key in seen for key in key_set):
            errors.append("duplicate")
        if errors:
            rejected.append(item)
            status = "reject"
        else:
            audited.append(item)
            seen.update(key_set)
            status = "accept"
        records.append({"item_id": item.get("id"), "status": status, "reasons": errors})
        record_event(
            events_path,
            f"audit_{status}",
            run_id=run_id,
            dataset_id=item.get("dataset"),
            task_id=item.get("task_id"),
            item_id=item.get("id"),
            agent="audit",
            operation="deterministic",
            status=status,
        )
    output = Path(output_directory)
    prefix = f"benchmark.{blueprint['dataset']['dataset_id']}"
    balanced = balance_items(audited, seed)
    audited_path, flagged_path, rejected_path = (
        output / f"{prefix}.audited.jsonl",
        output / f"{prefix}.flagged.jsonl",
        output / f"{prefix}.rejected.jsonl",
    )
    records_path = output / "audit_records.jsonl"
    report_path = output / "audit_report.json"
    _write(audited_path, balanced)
    _write(flagged_path, [])
    _write(rejected_path, rejected)
    _write(records_path, records)
    report = {
        "input": len(candidates),
        "pass": len(audited),
        "flag": 0,
        "reject": len(rejected),
        "retention": len(audited) / len(candidates) if candidates else None,
        "reason_counts": dict(
            Counter(reason for record in records for reason in record["reasons"])
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {
        "audited_path": str(audited_path),
        "flagged_path": str(flagged_path),
        "rejected_path": str(rejected_path),
        "records_path": str(records_path),
        "report_path": str(report_path),
        "item_count": len(audited),
    }
