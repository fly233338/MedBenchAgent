"""Resumable model-driven benchmark generation from approved Blueprint tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.contracts import BenchmarkItem, BlueprintTask, GenerationRequest, GenerationResult
from medbenchagent.llm import ModelUnavailable, complete_json
from medbenchagent.sample_counts import distribute_sample_counts

DEFAULT_MAX_TASKS_PER_IMAGE = 3
DEFAULT_QUESTION_TEMPERATURE = 0.7
FETCH_BATCH_SIZE = 128
_MISSING = object()

VQA_GENERATION_PROMPT = """
Generate one letter-choice question for a medical image VQA benchmark.

Context:
- Language: {language}
- Modality: {modality}
- Task ID: {task_id}
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Canonical question: {canonical_question}
- Candidate options: {candidate_options}
- Supporting annotation: {supporting_annotation}

Requirements:
1. Write one substantial semantic paraphrase of the canonical question, not a trivial word substitution.
2. The question must preserve the task meaning and be answerable from the image and candidate options only.
3. Do not mention reports, annotations, labels, or unsupported clinical details.
4. Do not reveal or hint at the correct answer.
5. Normalize the annotation-supported answer to one exact candidate option; do not infer beyond the annotation.

Return JSON only:
{{"question": "question stem", "answer": "one exact candidate option"}}
""".strip()

VQA_REVIEW_PROMPT = """
Review one generated letter-choice VQA item.

Context:
- Task definition: {task_definition}
- Generation logic: {generation_logic}
- Candidate options: {candidate_options}
- Supporting annotation: {supporting_annotation}
- Question: {question}
- Answer: {answer}

Approve only when:
1. The question does not reveal or clearly hint at the answer.
2. The answer is exactly supported by the annotation.

Return JSON only:
{{"approved": true, "reason": "short reason"}}
""".strip()


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


def _normalized_options(options: object) -> list[str]:
    if not isinstance(options, list):
        return []
    return [re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", option).strip() if isinstance(option, str) else "" for option in options]


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
        if not isinstance(templates, list) or len(templates) != 1 or not all(
            isinstance(template, str) and template.strip() for template in templates
        ):
            raise ValueError(f"{task_id}: question_templates must contain exactly one non-empty string.")
        for field in ("generation_logic", "medical_task_description"):
            if not isinstance(task.get(field), str) or not str(task[field]).strip():
                raise ValueError(f"{task_id}: {field} must be a non-empty string.")
        options = _normalized_options(task.get("options"))
        if response_format == "letter":
            if (
                not 2 <= len(options) <= 26
                or not all(options)
                or len(set(options)) != len(options)
            ):
                raise ValueError(f"{task_id}: letter tasks need 2-26 unique options.")
            task["options"] = options
        elif task.get("options") is not None:
            raise ValueError(f"{task_id}: options are allowed only for letter tasks.")
    max_tasks = request.get("max_tasks_per_image", DEFAULT_MAX_TASKS_PER_IMAGE)
    if not isinstance(max_tasks, int) or isinstance(max_tasks, bool) or max_tasks <= 0:
        raise ValueError("max_tasks_per_image must be a positive integer.")
    temperature = request.get("question_temperature", DEFAULT_QUESTION_TEMPERATURE)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
        raise ValueError("question_temperature must be between 0 and 2.")
    return tasks, targets


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


def _task_definition(candidate: _Candidate) -> str:
    return json.dumps(
        {
            "description": candidate.task["description"],
            "require": candidate.task["require"],
        },
        ensure_ascii=False,
    )


def _letter_generation_prompt(candidate: _Candidate, benchmark_info: dict[str, object]) -> str:
    return VQA_GENERATION_PROMPT.format(
        language=benchmark_info.get("language", ""),
        modality=benchmark_info.get("modality", ""),
        task_id=candidate.task_id,
        task_definition=_task_definition(candidate),
        generation_logic=candidate.task["generation_logic"],
        canonical_question=candidate.task["question_templates"][0],
        candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
        supporting_annotation=json.dumps(candidate.supporting_annotation, ensure_ascii=False),
    )


def _validate_letter_response(candidate: _Candidate, response: object) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != {"question", "answer"}:
        raise ValueError("Letter generation must contain only question and answer.")
    question = response["question"]
    answer = response["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Generated question must be a non-empty string.")
    if not isinstance(answer, str) or answer not in candidate.task["options"]:
        raise ValueError("Letter answer must be one exact option.")
    return {"question": question.strip(), "answer": answer}


def _legacy_generation_prompt(candidate: _Candidate, benchmark_info: dict[str, object]) -> tuple[str, str]:
    system = (
        "Generate one medical benchmark item. Return exactly "
        '{"question":"...","answer":...,"source_refs":[{"field":"approved field",'
        '"instance_index":null}]}. Use only the supplied task and supporting annotation. Do not reveal '
        "the answer or invent facts. For bbox tasks copy one valid xyxy box and identify its object-array "
        "source index. For free text copy or organize only supplied evidence."
    )
    user = (
        f"Benchmark: {json.dumps(benchmark_info, ensure_ascii=False)}\n"
        f"Task ID: {candidate.task_id}\nTask: {json.dumps(candidate.task, ensure_ascii=False)}\n"
        f"Supporting annotation: {json.dumps(candidate.supporting_annotation, ensure_ascii=False)}"
    )
    return system, user


def _generate_candidate(
    candidate: _Candidate, benchmark_info: dict[str, object], question_temperature: float
) -> dict[str, Any]:
    if candidate.task["response_format"] == "letter":
        system = _letter_generation_prompt(candidate, benchmark_info)
        try:
            response = complete_json(system, "", temperature=question_temperature)
            return _validate_letter_response(candidate, response)
        except ValueError as first_error:
            repaired = complete_json(
                "Repair one letter-choice VQA response. Return JSON only.\n"
                f"{system}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
                f"Contract error: {first_error}",
                "",
                temperature=question_temperature,
            )
            return _validate_letter_response(candidate, repaired)

    system, user = _legacy_generation_prompt(candidate, benchmark_info)
    try:
        response = complete_json(system, user)
        return _validate_generated(candidate, response)
    except ValueError as first_error:
        repaired = complete_json(
            "Repair one benchmark generation response. Return JSON only.",
            f"{user}\nPrevious response: {json.dumps(response, ensure_ascii=False)}\n"
            f"Contract error: {first_error}\nRequired contract: {system}",
        )
        return _validate_generated(candidate, repaired)


def _review(
    candidate: _Candidate,
    generated: dict[str, Any],
    benchmark_info: dict[str, object],
) -> tuple[bool, str]:
    if candidate.task["response_format"] == "bbox":
        return _review_bbox(candidate, generated)
    if candidate.task["response_format"] == "letter":
        response = complete_json(
            VQA_REVIEW_PROMPT.format(
                task_definition=_task_definition(candidate),
                generation_logic=candidate.task["generation_logic"],
                candidate_options=json.dumps(candidate.task["options"], ensure_ascii=False),
                supporting_annotation=json.dumps(candidate.supporting_annotation, ensure_ascii=False),
                question=generated["question"],
                answer=generated["answer"],
            ),
            "",
        )
        approved = (
            isinstance(response, dict)
            and set(response) == {"approved", "reason"}
            and response.get("approved") is True
            and isinstance(response.get("reason"), str)
        )
        reason = response.get("reason") if isinstance(response, dict) else "Invalid review response."
        return approved, str(reason)
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
    item: BenchmarkItem = {
        "id": _item_identity(candidate),
        "task_id": candidate.task_id,
        "image_id": candidate.record["image_id"],
        "image_path": candidate.record["image_path"],
        "question": generated["question"],
        "options": None,
        "answer": generated["answer"],
    }
    if response_format == "letter":
        answer_value = generated["answer"]
        options = list(candidate.task["options"])
        assert isinstance(answer_value, str)
        item["options"] = options
        item["answer"] = chr(ord("A") + options.index(answer_value))
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
            str(item["image_id"]),
            str(item["id"]),
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


def _has_meaningful_value(value: object) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, dict):
        return any(key != "_source_index" and _has_meaningful_value(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_meaningful_value(item) for item in value)
    return True


def _shuffle_candidates(candidates: list[_Candidate], seed: int, image_id: str) -> None:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    random.Random(int.from_bytes(digest, "big")).shuffle(candidates)


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
                    if _has_meaningful_value(projected):
                        candidates.append(
                            _Candidate(task_id, record["_source_row"], record, projected, task)
                        )
                if candidates:
                    _shuffle_candidates(candidates, seed, str(record["image_id"]))
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

    accepted: list[BenchmarkItem] = []
    max_tasks = int(request.get("max_tasks_per_image", DEFAULT_MAX_TASKS_PER_IMAGE))
    temperature = float(request.get("question_temperature", DEFAULT_QUESTION_TEMPERATURE))
    for candidate in candidates:
        if len(accepted) >= max_tasks:
            break
        try:
            generated = _generate_candidate(candidate, request["benchmark_info"], temperature)
        except ModelUnavailable:
            rejections["generation_model_unavailable"] = rejections.get(
                "generation_model_unavailable", 0
            ) + 1
            continue
        except ValueError:
            rejections["generation_contract"] = rejections.get("generation_contract", 0) + 1
            continue
        try:
            approved, _reason = _review(candidate, generated, request["benchmark_info"])
        except ModelUnavailable:
            rejections["review_model_unavailable"] = rejections.get("review_model_unavailable", 0) + 1
            continue
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
