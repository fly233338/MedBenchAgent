"""Model-driven benchmark generation from Blueprint tasks and approved annotations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.contracts import BenchmarkItem, BlueprintTask, GenerationRequest, GenerationResult
from medbenchagent.llm import ModelUnavailable, complete_json

MAX_ATTEMPTS_PER_CANDIDATE = 3
_MISSING = object()


@dataclass(frozen=True)
class _Candidate:
    task_id: str
    source_row: int
    record: dict[str, Any]
    supporting_annotation: dict[str, Any]
    task: BlueprintTask

    @property
    def key(self) -> tuple[str, int]:
        return self.task_id, self.source_row


def _merge_projection(left: object, right: object) -> object:
    if isinstance(left, dict) and isinstance(right, dict):
        merged = copy.deepcopy(left)
        for key, value in right.items():
            merged[key] = _merge_projection(merged[key], value) if key in merged else copy.deepcopy(value)
        return merged
    if isinstance(left, list) and isinstance(right, list):
        if all(isinstance(item, dict) and "_source_index" in item for item in left + right):
            by_index = {int(item["_source_index"]): copy.deepcopy(item) for item in left}
            for item in right:
                index = int(item["_source_index"])
                by_index[index] = (
                    _merge_projection(by_index[index], item) if index in by_index else copy.deepcopy(item)
                )
            return [by_index[index] for index in sorted(by_index)]
    return copy.deepcopy(right)


def _project_value(value: object, segments: list[str]) -> object:
    if not segments:
        return copy.deepcopy(value)
    if not isinstance(value, dict):
        return _MISSING
    segment = segments[0]
    is_array = segment.endswith("[*]")
    key = segment.removesuffix("[*]")
    if key not in value:
        return _MISSING
    nested = value[key]
    if is_array:
        if not isinstance(nested, list):
            return _MISSING
        if len(segments) == 1:
            return {key: copy.deepcopy(nested)}
        projected_items: list[dict[str, Any]] = []
        for index, item in enumerate(nested):
            projected = _project_value(item, segments[1:])
            if projected is _MISSING:
                continue
            if isinstance(projected, dict):
                projected_items.append({"_source_index": index, **projected})
            else:
                projected_items.append({"_source_index": index, "value": projected})
        return {key: projected_items} if projected_items else _MISSING
    projected = _project_value(nested, segments[1:])
    return {key: projected} if projected is not _MISSING else _MISSING


def project_supporting_annotation(
    annotation: dict[str, Any], supporting_fields: list[str]
) -> dict[str, Any]:
    """Project approved annotation paths while retaining array source indices."""
    projected: object = {}
    for path in supporting_fields:
        if not isinstance(path, str) or not path.startswith("annotation."):
            raise ValueError(f"Supporting field must start with annotation.: {path}")
        value = _project_value(annotation, path.split(".")[1:])
        if value is not _MISSING:
            projected = _merge_projection(projected, value)
    return projected if isinstance(projected, dict) else {}


def _validate_tasks(request: GenerationRequest) -> dict[str, BlueprintTask]:
    tasks = request.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("No Blueprint task is available for generation.")
    for task_id, task in tasks.items():
        if not isinstance(task_id, str) or not task_id or not isinstance(task, dict):
            raise ValueError("Blueprint tasks must use non-empty string IDs and object values.")
        response_format = task.get("response_format")
        if response_format not in {"letter", "bbox", "free_text"}:
            raise ValueError(f"{task_id}: unsupported response_format {response_format}.")
        supporting = task.get("supporting_fields")
        if not isinstance(supporting, list) or not supporting or not all(
            isinstance(field, str) and field.startswith("annotation.") for field in supporting
        ):
            raise ValueError(f"{task_id}: supporting_fields must contain annotation paths.")
        templates = task.get("question_templates")
        if not isinstance(templates, list) or not templates or not all(
            isinstance(template, str) and template.strip() for template in templates
        ):
            raise ValueError(f"{task_id}: question_templates must be non-empty strings.")
        for field in ("generation_logic", "medical_task_description"):
            if not isinstance(task.get(field), str) or not str(task[field]).strip():
                raise ValueError(f"{task_id}: {field} must be a non-empty string.")
        options = task.get("options")
        if response_format == "letter":
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 26
                or not all(isinstance(option, str) and option for option in options)
                or len(set(options)) != len(options)
            ):
                raise ValueError(f"{task_id}: letter tasks need 2-26 unique options.")
        elif options is not None:
            raise ValueError(f"{task_id}: options are allowed only for letter tasks.")
    return tasks


def _load_records(request: GenerationRequest) -> list[dict[str, Any]]:
    database = request["dataset_access_manifest"].get("duckdb_path")
    if not isinstance(database, str):
        raise ValueError("Dataset access manifest lacks duckdb_path.")
    connection = duckdb.connect(database, read_only=True)
    try:
        rows = connection.execute(
            "SELECT image_id, image_path, patient_id, case_id, study_id, annotation, _source_row "
            "FROM images ORDER BY _source_row"
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "image_id": row[0],
            "image_path": row[1],
            "patient_id": row[2],
            "case_id": row[3],
            "study_id": row[4],
            "annotation": json.loads(row[5]) if isinstance(row[5], str) else row[5],
            "_source_row": int(row[6]),
        }
        for row in rows
    ]


def _load_candidates(
    request: GenerationRequest, tasks: dict[str, BlueprintTask]
) -> dict[str, list[_Candidate]]:
    candidates: dict[str, list[_Candidate]] = defaultdict(list)
    for record in _load_records(request):
        for task_id, task in tasks.items():
            projected = project_supporting_annotation(record["annotation"], task["supporting_fields"])
            if projected:
                candidates[task_id].append(
                    _Candidate(task_id, record["_source_row"], record, projected, task)
                )
    return candidates


def _answer_schema(response_format: str) -> str:
    if response_format == "letter":
        return '"one exact option string"'
    if response_format == "bbox":
        return "[x1,y1,x2,y2]"
    return '"non-empty reference text"'


def _has_source_index(value: object, index: int) -> bool:
    if isinstance(value, dict):
        if value.get("_source_index") == index:
            return True
        return any(_has_source_index(nested, index) for nested in value.values())
    if isinstance(value, list):
        return any(_has_source_index(nested, index) for nested in value)
    return False


def _generation_prompt(candidate: _Candidate, benchmark_info: dict[str, object]) -> str:
    task = candidate.task
    response_format = task["response_format"]
    return (
        "Generate exactly one benchmark question and its answer from approved supporting annotation. "
        "Return either "
        '{"status":"generated","question":"...","answer":'
        f"{_answer_schema(response_format)},"
        '"source_refs":[{"field":"one approved supporting field","instance_index":null}]} '
        'or {"status":"skip","reason":"..."}. '
        "Follow the task generation_logic and question templates. Use only the supplied supporting "
        "annotation, never invent facts, and do not reveal the answer in the question. For letter tasks, "
        "normalize the annotation to exactly one supplied option. For bbox tasks, copy one valid xyxy "
        "box associated with the target described by the question. For free-text tasks, copy or organize "
        "only supplied annotation content. source_refs must identify the approved paths used; replace "
        "instance_index null with the non-negative source index when the field belongs to an object array.\n\n"
        f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"Task ID: {candidate.task_id}\n"
        f"Task: {json.dumps(task, ensure_ascii=False)}\n"
        f"Supporting annotation: {json.dumps(candidate.supporting_annotation, ensure_ascii=False)}"
    )


def _validate_generated(candidate: _Candidate, response: object) -> dict[str, Any] | None:
    if not isinstance(response, dict) or response.get("status") not in {"generated", "skip"}:
        raise ValueError("Generation response must have status generated or skip.")
    if response["status"] == "skip":
        if set(response) != {"status", "reason"} or not isinstance(response.get("reason"), str) or not response["reason"].strip():
            raise ValueError("Skipped generation must contain one non-empty reason.")
        return None
    if set(response) != {"status", "question", "answer", "source_refs"}:
        raise ValueError("Generated response has unexpected fields.")
    question = response["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    refs = response["source_refs"]
    supporting = set(candidate.task["supporting_fields"])
    if not isinstance(refs, list) or not refs:
        raise ValueError("Generated source_refs must be a non-empty list.")
    normalized_refs: list[dict[str, object]] = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"field", "instance_index"}:
            raise ValueError("Each source ref must contain field and instance_index.")
        field = ref["field"]
        index = ref["instance_index"]
        if field not in supporting or (index is not None and (not isinstance(index, int) or index < 0)):
            raise ValueError("Source refs must use approved fields and non-negative indices.")
        field_projection = _project_value(candidate.record["annotation"], field.split(".")[1:])
        if field_projection is _MISSING:
            raise ValueError("Source refs must identify fields present in this annotation.")
        has_parent_array = any(segment.endswith("[*]") for segment in field.split(".")[1:-1])
        if has_parent_array and index is None:
            raise ValueError("Source refs for object-array fields require instance_index.")
        if isinstance(index, int) and not _has_source_index(field_projection, index):
            raise ValueError("Source ref instance_index does not exist in supporting annotation.")
        normalized_refs.append({"field": field, "instance_index": index})
    answer = response["answer"]
    response_format = candidate.task["response_format"]
    if response_format == "letter":
        if not isinstance(answer, str) or answer not in candidate.task["options"]:
            raise ValueError("Letter answer must be one exact option.")
        normalized_answer: str | list[float] = answer
    elif response_format == "bbox":
        if (
            not isinstance(answer, list)
            or len(answer) != 4
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in answer)
        ):
            raise ValueError("Bbox answer must contain four numbers.")
        normalized_answer = [float(value) for value in answer]
        if normalized_answer[2] <= normalized_answer[0] or normalized_answer[3] <= normalized_answer[1]:
            raise ValueError("Bbox answer must be valid xyxy coordinates.")
    else:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Free-text answer must be a non-empty string.")
        normalized_answer = answer.strip()
    return {
        "question": question.strip(),
        "answer": normalized_answer,
        "source_refs": normalized_refs,
    }


def _review(
    candidate: _Candidate,
    generated: dict[str, Any],
    benchmark_info: dict[str, object],
) -> tuple[bool, str]:
    try:
        response = complete_json(
            "Independently audit one generated medical benchmark item. Return exactly "
            '{"approved":true,"reason":"..."}, using false when rejected. Approve only when the question follows the '
            "task templates and generation logic, the answer is fully supported by the supplied annotation, "
            "normalization is medically faithful, bbox target and coordinates refer to the same finding, "
            "no unsupported content is added, and the question does not reveal its answer.",
            f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
            f"Task: {json.dumps(candidate.task, ensure_ascii=False)}\n"
            f"Supporting annotation: {json.dumps(candidate.supporting_annotation, ensure_ascii=False)}\n"
            f"Generated item: {json.dumps(generated, ensure_ascii=False)}",
        )
    except ModelUnavailable:
        return False, "Review model was unavailable."
    approved = (
        isinstance(response, dict)
        and set(response) == {"approved", "reason"}
        and response.get("approved") is True
        and isinstance(response.get("reason"), str)
    )
    reason = response.get("reason") if isinstance(response, dict) else "Invalid review response."
    return approved, str(reason)


def _generate_candidate(
    candidate: _Candidate, benchmark_info: dict[str, object]
) -> dict[str, Any] | None:
    prompt = _generation_prompt(candidate, benchmark_info)
    retry_feedback = ""
    for _attempt in range(MAX_ATTEMPTS_PER_CANDIDATE):
        try:
            response = complete_json(
                "Generate one benchmark question and answer using only approved annotation evidence.",
                prompt + retry_feedback,
            )
            generated = _validate_generated(candidate, response)
        except ModelUnavailable:
            retry_feedback = "\nPrevious attempt failed because the generation model was unavailable."
            continue
        except ValueError as exc:
            retry_feedback = f"\nPrevious response violated the output contract: {exc}"
            continue
        if generated is None:
            return None
        approved, reason = _review(candidate, generated, benchmark_info)
        if approved:
            return generated
        retry_feedback = (
            "\nPrevious generated item was rejected by the independent reviewer: " + reason
        )
    return None


def _item_identity(candidate: _Candidate) -> str:
    raw = f"{candidate.task_id}:{candidate.record['image_id']}:{candidate.source_row}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _build_item(candidate: _Candidate, generated: dict[str, Any]) -> BenchmarkItem:
    response_format = candidate.task["response_format"]
    refs = [
        {
            "image_id": candidate.record["image_id"],
            "source_row": candidate.source_row,
            **ref,
        }
        for ref in generated["source_refs"]
    ]
    item: BenchmarkItem = {
        "id": _item_identity(candidate),
        "task_id": candidate.task_id,
        "task_type": candidate.task["task_type"],
        "response_format": response_format,
        "image_ids": [candidate.record["image_id"]],
        "image_paths": [candidate.record["image_path"]],
        "question": generated["question"],
        "options": None,
        "answer": generated["answer"],
        "source_refs": refs,
        "medical_sources": list(candidate.task.get("citations", [])),
    }
    if response_format == "letter":
        answer_value = generated["answer"]
        options = list(candidate.task["options"])
        assert isinstance(answer_value, str)
        item["options"] = options
        item["answer"] = chr(ord("A") + options.index(answer_value))
        item["answer_value"] = answer_value
    return item


def _generate_from_candidates(
    candidates: list[_Candidate],
    needed: int,
    consumed: set[tuple[str, int]],
    benchmark_info: dict[str, object],
) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for candidate in candidates:
        if candidate.key in consumed:
            continue
        consumed.add(candidate.key)
        generated = _generate_candidate(candidate, benchmark_info)
        if generated is None:
            continue
        items.append(_build_item(candidate, generated))
        if len(items) == needed:
            break
    return items


def generate_raw_benchmark(request: GenerationRequest) -> GenerationResult:
    """Generate exactly target_sample_count reviewed benchmark items atomically."""
    target = request["target_sample_count"]
    if not isinstance(target, int) or target <= 0:
        raise ValueError("target_sample_count must be a positive integer.")
    tasks = _validate_tasks(request)
    candidates = _load_candidates(request, tasks)
    active_ids = [task_id for task_id in tasks if candidates[task_id]]
    if not active_ids or sum(len(candidates[task_id]) for task_id in active_ids) < target:
        raise ValueError("Insufficient valid candidates to reach target_sample_count.")

    quotas = {task_id: target // len(active_ids) for task_id in active_ids}
    for task_id in active_ids[: target % len(active_ids)]:
        quotas[task_id] += 1
    consumed: set[tuple[str, int]] = set()
    items: list[BenchmarkItem] = []
    benchmark_info = request["benchmark_info"]
    for task_id in active_ids:
        items.extend(
            _generate_from_candidates(candidates[task_id], quotas[task_id], consumed, benchmark_info)
        )
    shortfall = target - len(items)
    if shortfall:
        for task_id in active_ids:
            additions = _generate_from_candidates(
                candidates[task_id], shortfall, consumed, benchmark_info
            )
            items.extend(additions)
            shortfall -= len(additions)
            if not shortfall:
                break
    if shortfall:
        raise ValueError("Unable to generate target_sample_count reviewed benchmark items.")

    task_order = {task_id: index for index, task_id in enumerate(tasks)}
    items.sort(
        key=lambda item: (task_order[item["task_id"]], item["source_refs"][0]["source_row"])
    )
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
    counts = {task_id: sum(item["task_id"] == task_id for item in items) for task_id in tasks}
    return GenerationResult(str(output), len(items), counts)
