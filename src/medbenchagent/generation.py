"""Closed VQA generation from annotations, with model wording isolated from answers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.contracts import BenchmarkItem, GenerationRequest
from medbenchagent.llm import ModelUnavailable, complete_json

MAX_ATTEMPTS_PER_CANDIDATE = 3


@dataclass(frozen=True)
class GenerationResult:
    raw_output_path: str
    item_count: int
    task_counts: dict[str, int]


@dataclass(frozen=True)
class _Candidate:
    task_id: str
    source_row: int
    record: dict[str, Any]
    answer_value: str
    task: dict[str, Any]
    design: dict[str, Any]


def _path_value(annotation: dict[str, Any], path: str) -> object:
    if not path.startswith("annotation."):
        raise ValueError(f"Answer field must start with annotation.: {path}")
    values: list[object] = [annotation]
    for segment in path.split(".")[1:]:
        array = segment.endswith("[*]")
        key = segment.removesuffix("[*]")
        next_values: list[object] = []
        for value in values:
            if not isinstance(value, dict) or key not in value:
                continue
            nested = value[key]
            next_values.extend(nested if array and isinstance(nested, list) else [nested])
        values = next_values
    values = [value for value in values if value not in (None, "")]
    if len(values) != 1 or isinstance(values[0], (dict, list)):
        raise ValueError("Answer field must resolve to one scalar value.")
    return values[0]


def _task_config(request: GenerationRequest) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    designs = {str(design.get("task_id")): design for design in request["question_designs"]}
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task in request["closed_tasks"]:
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or task_id not in designs:
            raise ValueError("Each closed task must have a matching question design.")
        if task.get("task_type") != "close_vqa":
            raise ValueError(f"{task_id}: only close_vqa tasks may be generated.")
        options = task.get("options")
        answer_field = task.get("answer_field")
        if not isinstance(options, list) or not options or len(options) > 26 or not all(isinstance(value, str) for value in options):
            raise ValueError(f"{task_id}: fixed options must be 1–26 strings.")
        if len(set(options)) != len(options) or not isinstance(answer_field, str):
            raise ValueError(f"{task_id}: options must be unique and answer_field is required.")
        result.append((task, designs[task_id]))
    if not result:
        raise ValueError("No retained close_vqa task is available for generation.")
    return result


def _candidate_value(task: dict[str, Any], record: dict[str, Any]) -> str | None:
    try:
        value = _path_value(record["annotation"], task["answer_field"])
    except ValueError:
        return None
    mapping = task.get("answer_value_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"{task['task_id']}: answer_value_mapping must be an object.")
    answer_value = mapping.get(str(value), value)
    return answer_value if isinstance(answer_value, str) and answer_value in task["options"] else None


def _load_candidates(request: GenerationRequest, task_designs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, list[_Candidate]]:
    manifest = request["dataset_access_manifest"]
    database = manifest.get("duckdb_path")
    if not isinstance(database, str):
        raise ValueError("Dataset access manifest lacks duckdb_path.")
    connection = duckdb.connect(database, read_only=True)
    try:
        rows = connection.execute(
            "SELECT image_id, image_path, patient_id, case_id, study_id, annotation, _source_row FROM images ORDER BY _source_row"
        ).fetchall()
    finally:
        connection.close()
    candidates: dict[str, list[_Candidate]] = defaultdict(list)
    for row in rows:
        record = {
            "image_id": row[0], "image_path": row[1], "patient_id": row[2], "case_id": row[3], "study_id": row[4],
            "annotation": json.loads(row[5]) if isinstance(row[5], str) else row[5], "_source_row": int(row[6]),
        }
        for task, design in task_designs:
            if (answer_value := _candidate_value(task, record)) is not None:
                candidates[task["task_id"]].append(
                    _Candidate(task["task_id"], record["_source_row"], record, answer_value, task, design)
                )
    return candidates


def _question(candidate: _Candidate) -> str | None:
    wording_design = {
        key: value
        for key, value in candidate.design.items()
        if key not in {"answer_list", "answer_field", "answer_value_mapping", "source_value_mapping"}
    }
    prompt = (
        "Generate one concise closed medical VQA question. Return JSON only as {\"question\":\"...\"}. "
        "Do not state, alter, infer, or mention the answer, answer options, or label.\n\n"
        f"Task: {candidate.task['task_id']}\nDescription: {candidate.task.get('description', '')}\n"
        f"Question design: {json.dumps(wording_design, ensure_ascii=False)}"
    )
    try:
        response = complete_json("You write only the question wording for a benchmark item.", prompt)
    except ModelUnavailable:
        return None
    if set(response) != {"question"} or not isinstance(response["question"], str) or not response["question"].strip():
        return None
    return response["question"].strip()


def _approved(question: str, candidate: _Candidate) -> bool:
    try:
        response = complete_json(
            "Independently review benchmark question wording. Return JSON only as {\"approved\":true|false}.",
            f"Task: {candidate.task['task_id']}\nQuestion: {question}\n"
            "Approve only if it is a clear question and does not expose or change the answer.",
        )
    except ModelUnavailable:
        return False
    return set(response) == {"approved"} and response["approved"] is True


def _build_item(candidate: _Candidate, question: str) -> BenchmarkItem:
    options = list(candidate.task["options"])
    answer_index = options.index(candidate.answer_value)
    item_id = hashlib.sha256(f"{candidate.task_id}:{candidate.record['image_id']}".encode("utf-8")).hexdigest()[:24]
    return {
        "id": item_id,
        "task_id": candidate.task_id,
        "image_ids": [candidate.record["image_id"]],
        "image_paths": [candidate.record["image_path"]],
        "question": question,
        "options": options,
        "answer": chr(ord("A") + answer_index),
        "answer_value": candidate.answer_value,
        "source_refs": [{"image_id": candidate.record["image_id"], "source_row": candidate.source_row, "answer_field": candidate.task["answer_field"]}],
        "medical_sources": list(candidate.design.get("citations", [])),
    }


def _generate_from_candidates(candidates: list[_Candidate], needed: int, used: set[tuple[str, int]]) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for candidate in candidates:
        key = (candidate.task_id, candidate.source_row)
        if key in used:
            continue
        for _attempt in range(MAX_ATTEMPTS_PER_CANDIDATE):
            question = _question(candidate)
            if question and _approved(question, candidate):
                items.append(_build_item(candidate, question))
                used.add(key)
                break
        if len(items) == needed:
            return items
    return items


def generate_raw_benchmark(request: GenerationRequest) -> GenerationResult:
    """Generate exactly target_sample_count raw items or leave no partial output behind."""
    target = request["target_sample_count"]
    if not isinstance(target, int) or target <= 0:
        raise ValueError("target_sample_count must be a positive integer.")
    task_designs = _task_config(request)
    candidates = _load_candidates(request, task_designs)
    if sum(len(candidates[task["task_id"]]) for task, _design in task_designs) < target:
        raise ValueError("Insufficient valid candidates to reach target_sample_count.")
    quotas = {task["task_id"]: target // len(task_designs) for task, _design in task_designs}
    for task, _design in task_designs[: target % len(task_designs)]:
        quotas[task["task_id"]] += 1
    used: set[tuple[str, int]] = set()
    items: list[BenchmarkItem] = []
    for task, _design in task_designs:
        items.extend(_generate_from_candidates(candidates[task["task_id"]], quotas[task["task_id"]], used))
    shortfall = target - len(items)
    if shortfall:
        for task, _design in task_designs:
            additions = _generate_from_candidates(candidates[task["task_id"]], shortfall, used)
            items.extend(additions)
            shortfall -= len(additions)
            if not shortfall:
                break
    if shortfall:
        raise ValueError("Unable to generate target_sample_count validated benchmark items.")
    task_order = {task["task_id"]: index for index, (task, _design) in enumerate(task_designs)}
    items.sort(key=lambda item: (task_order[item["task_id"]], item["source_refs"][0]["source_row"]))
    output_dir = Path(request["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "benchmark_items.raw.jsonl"
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for item in items:
                destination.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    counts = {task_id: sum(item["task_id"] == task_id for item in items) for task_id in task_order}
    return GenerationResult(str(output), len(items), counts)
