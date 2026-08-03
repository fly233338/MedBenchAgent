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
from medbenchagent.construction import AUDIT_AGENT, record_event
from medbenchagent.dataset import load_annotation_jsonl
from medbenchagent.generation.context import (
    evidence_sha256,
    evidence_value,
    normalized_bbox,
)
from medbenchagent.generation.engine import project_supporting_record
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


def _meaningful(value: object) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, dict):
        return any(key != "_source_index" and _meaningful(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_meaningful(item) for item in value)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    if _sha256(image_path) != entry.get("sha256"):
        return ["image_hash"]
    try:
        with Image.open(image_path) as image:
            image.load()
            if (
                image.width <= 0
                or image.height <= 0
                or image.width != entry.get("width")
                or image.height != entry.get("height")
            ):
                return ["image_dimensions"]
            thumbnail = image.convert("L").resize((256, 256))
            minimum, maximum = thumbnail.getextrema()
            if minimum == maximum:
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
            or not (0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1])
        ):
            errors.append("bbox")
    else:
        errors.append("response_format")
    return errors


def _replay_witness(
    record: dict[str, object],
    supporting_fields: list[str],
    witness: object,
) -> tuple[str, list[int], object]:
    if not isinstance(witness, dict) or set(witness) != {
        "path",
        "indices",
        "evidence_sha256",
    }:
        raise ValueError("Evidence witness has an invalid structure.")
    path, indices = witness.get("path"), witness.get("indices")
    if not isinstance(path, str) or path not in supporting_fields:
        raise ValueError("Evidence path is not a supporting field.")
    if not isinstance(indices, list):
        raise ValueError("Evidence indices must be a list.")
    value = evidence_value(record, path, indices)
    if evidence_sha256(path, indices, value) != witness.get("evidence_sha256"):
        raise ValueError("Evidence hash mismatch.")
    return path, indices, value


def _answer_evidence_matches(
    item: dict[str, Any],
    task: dict[str, Any],
    source: dict[str, Any],
    record: dict[str, object],
    supporting_fields: list[str],
) -> bool:
    try:
        witness = source.get("answer_evidence")
        if task.get("response_format") != "bbox":
            path, _indices, value = _replay_witness(record, supporting_fields, witness)
            if task.get("response_format") == "letter":
                if not isinstance(task.get("answer_value_mapping"), list):
                    return False
                options, answer = item.get("options"), item.get("answer")
                if (
                    path != task.get("answer_field")
                    or not isinstance(options, list)
                    or not isinstance(answer, str)
                    or len(answer) != 1
                    or not 0 <= ord(answer.upper()) - ord("A") < len(options)
                ):
                    return False
                expected_option = options[ord(answer.upper()) - ord("A")]
                matches = [
                    mapping
                    for mapping in task["answer_value_mapping"]
                    if isinstance(mapping, dict)
                    and type(mapping.get("raw_value")) is type(value)
                    and mapping.get("raw_value") == value
                    and mapping.get("option") == expected_option
                ]
                return len(matches) == 1
            return task.get("response_format") != "free_text" or (
                isinstance(value, str) and value == item.get("answer")
            )
        if not isinstance(witness, list) or len(witness) != 4:
            return False
        replayed = [
            _replay_witness(record, supporting_fields, coordinate)
            for coordinate in witness
        ]
        paths = [path for path, _indices, _value in replayed]
        indices = [coordinate_indices for _path, coordinate_indices, _value in replayed]
        if (
            len(set(paths)) != 1
            or not paths[0].endswith("[*]")
            or any(not coordinate_indices for coordinate_indices in indices)
            or any(
                coordinate_indices[:-1] != indices[0][:-1]
                or coordinate_indices[-1] != coordinate
                for coordinate, coordinate_indices in enumerate(indices)
            )
        ):
            return False
        source_bbox = normalized_bbox([value for _path, _indices, value in replayed])
        return source_bbox is not None and source_bbox == normalized_bbox(item.get("answer"))
    except (TypeError, ValueError):
        return False


def audit_run(
    *,
    raw_path: str | Path,
    blueprint_path: str | Path,
    provenance_path: str | Path,
    events_path: str | Path,
    manifest_path: str | Path,
    annotation_path: str | Path,
    annotation_sha256: str,
    output_directory: str | Path,
    run_id: str,
    seed: int = 0,
) -> dict[str, str | int]:
    blueprint = json.loads(Path(blueprint_path).read_text(encoding="utf-8"))
    tasks = blueprint["tasks"]
    annotation_source = Path(annotation_path)
    annotation_hash_matches = _sha256(annotation_source) == annotation_sha256
    annotation_rows = {
        record["_source_row"]: record for record in load_annotation_jsonl(annotation_source)
    }
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
        if not annotation_hash_matches:
            errors.append("annotation_hash")
        if isinstance(source, dict):
            source_row = source.get("source_row")
            annotation_record = (
                annotation_rows.get(source_row)
                if isinstance(source_row, int) and not isinstance(source_row, bool)
                else None
            )
            if (
                not isinstance(annotation_record, dict)
                or not isinstance(annotation_record.get("image"), dict)
                or annotation_record["image"].get("image_id") != image_id
            ):
                errors.append("annotation_source")
            elif isinstance(task, dict):
                supporting_fields = task.get("supporting_fields")
                if (
                    not isinstance(supporting_fields, list)
                    or not all(
                        _meaningful(project_supporting_record(annotation_record, [field]))
                        for field in supporting_fields
                    )
                ):
                    errors.append("annotation_evidence")
                elif not _answer_evidence_matches(
                    item,
                    task,
                    source,
                    annotation_record,
                    supporting_fields,
                ):
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
        records.append({"item_id": item.get("id"), "status": status, "reasons": errors})
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
        "schema_version": "audit-report.v1",
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
        "flagged_path": str(flagged_path),
        "rejected_path": str(rejected_path),
        "records_path": str(records_path),
        "report_path": str(report_path),
        "item_count": len(audited),
    }
