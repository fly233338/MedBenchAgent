"""Resumable model-driven benchmark generation from approved Blueprint tasks."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.contracts import BenchmarkItem, BlueprintTask, GenerationRequest, GenerationResult
from medbenchagent.llm import ModelUnavailable, complete_json
from medbenchagent.sample_counts import distribute_sample_counts

MAX_ITEMS_PER_RECORD = 3
FETCH_BATCH_SIZE = 128
_MISSING = object()


@dataclass(frozen=True)
class _Candidate:
    task_id: str
    source_row: int
    record: dict[str, Any]
    supporting_annotation: dict[str, Any]
    task: BlueprintTask


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


def _task_targets(request: GenerationRequest) -> dict[str, int]:
    tasks = request.get("tasks")
    target = request.get("target_sample_count")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("No Blueprint task is available for generation.")
    if not isinstance(target, int) or target <= 0:
        raise ValueError("target_sample_count must be a positive integer.")
    raw_counts = [task.get("sample_count") for task in tasks.values() if isinstance(task, dict)]
    if len(raw_counts) != len(tasks):
        raise ValueError("Blueprint tasks must be objects.")
    if all(count is None for count in raw_counts):
        return distribute_sample_counts(list(tasks), target)
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in raw_counts
    ):
        raise ValueError("Every Blueprint task must have a non-negative integer sample_count.")
    counts = {task_id: int(task["sample_count"]) for task_id, task in tasks.items()}
    if sum(counts.values()) != target:
        raise ValueError("Task sample counts must sum to target_sample_count.")
    return counts


def _validate_tasks(request: GenerationRequest) -> tuple[dict[str, BlueprintTask], dict[str, int]]:
    tasks = request["tasks"]
    targets = _task_targets(request)
    for task_id, task in tasks.items():
        if targets[task_id] == 0:
            continue
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
    return tasks, targets


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


def _resolve_source_value(
    annotation: dict[str, Any], field: str, instance_index: int | None
) -> object:
    """Resolve one approved annotation field using its object-array instance index."""
    current: object = annotation
    parent_array_seen = False
    segments = field.split(".")[1:]
    for position, segment in enumerate(segments):
        if not isinstance(current, dict):
            return _MISSING
        is_array = segment.endswith("[*]")
        key = segment.removesuffix("[*]")
        if key not in current:
            return _MISSING
        current = current[key]
        if not is_array or position == len(segments) - 1:
            continue
        if (
            parent_array_seen
            or instance_index is None
            or not isinstance(current, list)
            or instance_index >= len(current)
        ):
            return _MISSING
        current = current[instance_index]
        parent_array_seen = True
    return current


def _normalized_bbox(value: object) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
            for coordinate in value
        )
    ):
        return None
    bbox = [float(coordinate) for coordinate in value]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def _review_bbox(candidate: _Candidate, generated: dict[str, Any]) -> tuple[bool, str]:
    referenced_boxes: list[list[float]] = []
    for ref in generated["source_refs"]:
        value = _resolve_source_value(
            candidate.record["annotation"],
            str(ref["field"]),
            ref["instance_index"] if isinstance(ref["instance_index"], int) else None,
        )
        bbox = _normalized_bbox(value)
        if bbox is not None:
            referenced_boxes.append(bbox)
    if len(referenced_boxes) != 1:
        return False, "Detection requires exactly one unambiguous bbox source reference."
    if referenced_boxes[0] != generated["answer"]:
        return False, "Detection answer does not match its referenced annotation bbox."
    return True, "Detection answer matches its referenced annotation bbox."


def _validate_generated(candidate: _Candidate, response: object) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"question", "answer", "source_refs"}:
        raise ValueError("Generated item must contain question, answer, and source_refs.")
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


def _routing_prompt(
    candidates: list[_Candidate], benchmark_info: dict[str, object], remaining: dict[str, int]
) -> tuple[str, str]:
    offered = [
        {
            "task_id": candidate.task_id,
            "remaining_count": remaining[candidate.task_id],
            "task": candidate.task,
            "supporting_annotation": candidate.supporting_annotation,
            "answer_schema": _answer_schema(candidate.task["response_format"]),
        }
        for candidate in candidates
    ]
    system = (
        "Select medically suitable tasks for one annotated image and generate benchmark items in one pass. "
        "Return exactly {\"items\":[{\"task_id\":\"offered task\",\"question\":\"...\","
        "\"answer\":...,\"source_refs\":[{\"field\":\"approved field\",\"instance_index\":null}]}]}. "
        f"Return at most {MAX_ITEMS_PER_RECORD} items with unique task_id values, or an empty items list. "
        "Use only each task's own supporting annotation and approved fields. Follow its templates and "
        "generation_logic, never invent facts, and do not reveal the answer in the question. For letter "
        "tasks use one exact supplied option. For bbox tasks copy one valid xyxy box and identify its "
        "object-array source index. For free text copy or organize only supplied evidence."
    )
    user = (
        f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"Image ID: {candidates[0].record['image_id']}\n"
        f"Offered tasks: {json.dumps(offered, ensure_ascii=False)}"
    )
    return system, user


def _validate_routed_response(
    candidates: list[_Candidate], response: object
) -> list[tuple[_Candidate, dict[str, Any]]]:
    if not isinstance(response, dict) or set(response) != {"items"}:
        raise ValueError("Routed generation response must contain only items.")
    items = response["items"]
    if not isinstance(items, list) or len(items) > MAX_ITEMS_PER_RECORD:
        raise ValueError(f"Routed generation must return at most {MAX_ITEMS_PER_RECORD} items.")
    by_task = {candidate.task_id: candidate for candidate in candidates}
    seen: set[str] = set()
    normalized: list[tuple[_Candidate, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"task_id", "question", "answer", "source_refs"}:
            raise ValueError("Every routed item must contain task_id, question, answer, and source_refs.")
        task_id = item["task_id"]
        if task_id not in by_task or task_id in seen:
            raise ValueError("Routed task IDs must be unique offered tasks.")
        seen.add(task_id)
        generated = _validate_generated(
            by_task[task_id], {key: value for key, value in item.items() if key != "task_id"}
        )
        normalized.append((by_task[task_id], generated))
    return normalized


def _review(
    candidate: _Candidate,
    generated: dict[str, Any],
    benchmark_info: dict[str, object],
) -> tuple[bool, str]:
    if candidate.task["response_format"] == "bbox":
        return _review_bbox(candidate, generated)
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for item in items:
                destination.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _progress_counts(items: list[dict[str, Any]], task_ids: list[str]) -> dict[str, int]:
    return {task_id: sum(item.get("task_id") == task_id for item in items) for task_id in task_ids}


def _merge_progress(
    path: Path,
    additions: list[BenchmarkItem],
    task_order: dict[str, int],
) -> list[dict[str, Any]]:
    by_id = {str(item["id"]): item for item in _read_jsonl(path)}
    by_id.update({item["id"]: item for item in additions})
    items = list(by_id.values())
    items.sort(
        key=lambda item: (
            task_order[str(item["task_id"])],
            int(item["source_refs"][0]["source_row"]),
        )
    )
    _write_jsonl(path, items)
    return items


def _row_record(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "image_id": row[0],
        "image_path": row[1],
        "patient_id": row[2],
        "case_id": row[3],
        "study_id": row[4],
        "annotation": json.loads(row[5]) if isinstance(row[5], str) else row[5],
        "_source_row": int(row[6]),
    }


def _next_candidate_record(
    request: GenerationRequest,
    tasks: dict[str, BlueprintTask],
    targets: dict[str, int],
    completed: dict[str, int],
    cursor: int,
) -> tuple[int, list[_Candidate] | None]:
    database = request["dataset_access_manifest"].get("duckdb_path")
    if not isinstance(database, str):
        raise ValueError("Dataset access manifest lacks duckdb_path.")
    seed = int(request.get("seed", 0))
    connection = duckdb.connect(database, read_only=True)
    try:
        while True:
            rows = connection.execute(
                "SELECT image_id, image_path, patient_id, case_id, study_id, annotation, _source_row "
                "FROM images ORDER BY hash(_source_row, ?), _source_row LIMIT ? OFFSET ?",
                [seed, FETCH_BATCH_SIZE, cursor],
            ).fetchall()
            if not rows:
                return cursor, None
            for row in rows:
                cursor += 1
                record = _row_record(row)
                candidates: list[_Candidate] = []
                for task_id, task in tasks.items():
                    if completed[task_id] >= targets[task_id]:
                        continue
                    projected = project_supporting_annotation(
                        record["annotation"], task["supporting_fields"]
                    )
                    if projected:
                        candidates.append(
                            _Candidate(task_id, record["_source_row"], record, projected, task)
                        )
                if candidates:
                    candidates.sort(
                        key=lambda candidate: (
                            -(targets[candidate.task_id] - completed[candidate.task_id]),
                            list(tasks).index(candidate.task_id),
                        )
                    )
                    return cursor, candidates
            if len(rows) < FETCH_BATCH_SIZE:
                return cursor, None
    finally:
        connection.close()


def initialize_generation(request: GenerationRequest) -> dict[str, Any]:
    """Validate a request and initialize compact resumable generation state."""
    tasks, targets = _validate_tasks(request)
    output_dir = Path(request["output_directory"])
    progress_path = output_dir / "benchmark_items.progress.jsonl"
    existing = _read_jsonl(progress_path)
    completed = _progress_counts(existing, list(tasks))
    return {
        "generation_target_counts": targets,
        "generation_completed_counts": completed,
        "generation_cursor": 0,
        "generation_progress_path": str(progress_path),
        "generation_rejections": {},
        "generation_exhausted": False,
    }


def generation_done(state: dict[str, Any]) -> bool:
    targets = state["generation_target_counts"]
    completed = state["generation_completed_counts"]
    return bool(state.get("generation_exhausted")) or all(
        completed[task_id] >= target for task_id, target in targets.items()
    )


def advance_generation(request: GenerationRequest, state: dict[str, Any]) -> dict[str, Any]:
    """Process one model-worthy database record and return the next checkpoint state."""
    tasks, targets = _validate_tasks(request)
    progress_path = Path(state["generation_progress_path"])
    progress = _read_jsonl(progress_path)
    completed = _progress_counts(progress, list(tasks))
    cursor, candidates = _next_candidate_record(
        request, tasks, targets, completed, int(state["generation_cursor"])
    )
    rejections = dict(state.get("generation_rejections", {}))
    if candidates is None:
        return {
            "generation_cursor": cursor,
            "generation_completed_counts": completed,
            "generation_rejections": rejections,
            "generation_exhausted": True,
        }

    remaining = {task_id: targets[task_id] - completed[task_id] for task_id in tasks}
    system, user = _routing_prompt(candidates, request["benchmark_info"], remaining)
    try:
        response = complete_json(system, user)
    except ModelUnavailable:
        rejections["generation_model_unavailable"] = rejections.get(
            "generation_model_unavailable", 0
        ) + 1
        return {
            "generation_cursor": cursor,
            "generation_completed_counts": completed,
            "generation_rejections": rejections,
        }
    try:
        routed = _validate_routed_response(candidates, response)
    except ValueError as first_error:
        try:
            repaired = complete_json(
                "Repair one routed benchmark generation response. Return JSON only.",
                f"{user}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}\nRequired contract: {system}",
            )
            routed = _validate_routed_response(candidates, repaired)
        except (ModelUnavailable, ValueError):
            rejections["generation_contract"] = rejections.get("generation_contract", 0) + 1
            return {
                "generation_cursor": cursor,
                "generation_completed_counts": completed,
                "generation_rejections": rejections,
            }

    if not routed:
        rejections["model_skip"] = rejections.get("model_skip", 0) + 1
    accepted: list[BenchmarkItem] = []
    for candidate, generated in routed:
        approved, _reason = _review(candidate, generated, request["benchmark_info"])
        if approved:
            accepted.append(_build_item(candidate, generated))
        else:
            rejections["review_rejected"] = rejections.get("review_rejected", 0) + 1
    if accepted:
        progress = _merge_progress(
            progress_path, accepted, {task_id: index for index, task_id in enumerate(tasks)}
        )
        completed = _progress_counts(progress, list(tasks))
    return {
        "generation_cursor": cursor,
        "generation_completed_counts": completed,
        "generation_rejections": rejections,
    }


def finalize_generation(request: GenerationRequest, state: dict[str, Any]) -> GenerationResult:
    """Materialize the immutable raw JSONL from resumable progress."""
    tasks, targets = _validate_tasks(request)
    progress = _read_jsonl(Path(state["generation_progress_path"]))
    kept: list[dict[str, Any]] = []
    counts = {task_id: 0 for task_id in tasks}
    for item in progress:
        task_id = str(item.get("task_id"))
        if task_id in targets and counts[task_id] < targets[task_id]:
            kept.append(item)
            counts[task_id] += 1
    output = Path(request["output_directory"]) / "benchmark_items.raw.jsonl"
    _write_jsonl(output, kept)
    return GenerationResult(str(output), len(kept), counts)


def generate_raw_benchmark(request: GenerationRequest) -> GenerationResult:
    """Run the resumable generator synchronously for direct callers."""
    state = initialize_generation(request)
    while not generation_done(state):
        state.update(advance_generation(request, state))
    return finalize_generation(request, state)
