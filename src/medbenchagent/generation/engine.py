"""Shared resumable orchestration for staged benchmark generation."""

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

from medbenchagent.generation import free_text, grounding, vqa
from medbenchagent.contracts import BenchmarkItem, BlueprintTask, GenerationRequest, GenerationResult
from medbenchagent.construction import GENERATION_AGENT, event_context, record_event
from medbenchagent.llm import ModelUnavailable
from medbenchagent.sample_counts import distribute_sample_counts
from medbenchagent.settings import get_settings

_MISSING = object()
_STAGES = (
    ("vqa", "letter", vqa),
    ("grounding", "bbox", grounding),
    ("free_text", "free_text", free_text),
)


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
    return [
        re.sub(r"^\s*[A-Za-z]\s*[.)]\s*", "", option).strip()
        if isinstance(option, str)
        else ""
        for option in options
    ]


def _validate_tasks(request: GenerationRequest) -> tuple[dict[str, BlueprintTask], dict[str, int]]:
    dataset_id = request.get("dataset_id")
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", dataset_id):
        raise ValueError("dataset_id must be a stable lowercase slug.")
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
            if not 2 <= len(options) <= 26 or not all(options) or len(set(options)) != len(options):
                raise ValueError(f"{task_id}: letter tasks need 2-26 unique options.")
            task["options"] = options
        elif task.get("options") is not None:
            raise ValueError(f"{task_id}: options are allowed only for letter tasks.")
    max_tasks = request.get("max_tasks_per_image")
    if not isinstance(max_tasks, int) or isinstance(max_tasks, bool) or max_tasks <= 0:
        raise ValueError("max_tasks_per_image must be a positive integer.")
    temperature = request.get("question_temperature")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("question_temperature must be between 0 and 2.")
    return tasks, targets


def _item_identity(candidate: _Candidate) -> str:
    raw = f"{candidate.task_id}:{candidate.record['image_id']}:{candidate.source_row}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _build_item(
    candidate: _Candidate, generated: dict[str, Any], handler: Any, dataset_id: str
) -> BenchmarkItem:
    options, answer = handler.encode_answer(candidate, generated)
    return {
        "id": _item_identity(candidate),
        "task_id": candidate.task_id,
        "dataset": dataset_id,
        "image_id": candidate.record["image_id"],
        "image_path": candidate.record["image_path"],
        "question": generated["question"],
        "options": options,
        "answer": answer,
    }


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


def _progress_item(entry: dict[str, Any]) -> dict[str, Any]:
    item = entry.get("item", entry)
    return item if isinstance(item, dict) else {}


def _progress_counts(items: list[dict[str, Any]], task_ids: list[str]) -> dict[str, int]:
    return {
        task_id: sum(_progress_item(item).get("task_id") == task_id for item in items)
        for task_id in task_ids
    }


def _merge_progress(path: Path, additions: list[dict[str, Any]], task_order: dict[str, int]) -> list[dict[str, Any]]:
    by_id = {str(_progress_item(item)["id"]): item for item in _read_jsonl(path)}
    by_id.update({_progress_item(item)["id"]: item for item in additions})
    items = list(by_id.values())
    items.sort(
        key=lambda item: (
            task_order[str(_progress_item(item)["task_id"])],
            str(_progress_item(item)["image_id"]),
            str(_progress_item(item)["id"]),
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


def _all_supporting_fields_present(annotation: dict[str, Any], supporting_fields: list[str]) -> bool:
    return all(
        _has_meaningful_value(project_supporting_annotation(annotation, [field]))
        for field in supporting_fields
    )


def _next_candidate_record(
    request: GenerationRequest,
    tasks: dict[str, BlueprintTask],
    targets: dict[str, int],
    completed: dict[str, int],
    cursor: int,
    handler: Any,
) -> tuple[int, list[_Candidate] | None]:
    database = request["dataset_access_manifest"].get("duckdb_path")
    if not isinstance(database, str):
        raise ValueError("Dataset access manifest lacks duckdb_path.")
    seed = int(request.get("seed", get_settings().generation.seed))
    fetch_batch_size = get_settings().generation.fetch_batch_size
    connection = duckdb.connect(database, read_only=True)
    try:
        while True:
            rows = connection.execute(
                "SELECT image_id, image_path, patient_id, case_id, study_id, annotation, _source_row "
                "FROM images ORDER BY hash(_source_row, ?), _source_row LIMIT ? OFFSET ?",
                [seed, fetch_batch_size, cursor],
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
                    if not _all_supporting_fields_present(record["annotation"], task["supporting_fields"]):
                        continue
                    projected = project_supporting_annotation(record["annotation"], task["supporting_fields"])
                    candidate = _Candidate(task_id, record["_source_row"], record, projected, task)
                    if _has_meaningful_value(projected) and handler.supports(candidate):
                        candidates.append(candidate)
                if candidates:
                    _shuffle_candidates(candidates, seed, str(record["image_id"]))
                    return cursor, candidates
            if len(rows) < fetch_batch_size:
                return cursor, None
    finally:
        connection.close()


def _stage_progress_path(output_dir: Path, phase: str) -> Path:
    return output_dir / f"benchmark_items.{phase}.progress.jsonl"


def _stage_tasks(
    tasks: dict[str, BlueprintTask], targets: dict[str, int], response_format: str
) -> dict[str, BlueprintTask]:
    return {
        task_id: task
        for task_id, task in tasks.items()
        if task["response_format"] == response_format and targets[task_id] > 0
    }


def _all_completed_counts(
    output_dir: Path, tasks: dict[str, BlueprintTask], targets: dict[str, int]
) -> dict[str, int]:
    counts = {task_id: 0 for task_id in tasks}
    for phase, response_format, _handler in _STAGES:
        stage_tasks = _stage_tasks(tasks, targets, response_format)
        for task_id, count in _progress_counts(
            _read_jsonl(_stage_progress_path(output_dir, phase)), list(stage_tasks)
        ).items():
            counts[task_id] = count
    return counts


def _enter_stage(
    request: GenerationRequest,
    tasks: dict[str, BlueprintTask],
    targets: dict[str, int],
    completed: dict[str, int],
    stage_index: int,
    rejections: dict[str, int],
) -> dict[str, Any]:
    output_dir = Path(request["output_directory"])
    while stage_index < len(_STAGES):
        phase, response_format, _handler = _STAGES[stage_index]
        stage_tasks = _stage_tasks(tasks, targets, response_format)
        if stage_tasks and not all(completed[task_id] >= targets[task_id] for task_id in stage_tasks):
            return {
                "generation_stage_index": stage_index,
                "generation_stage": phase,
                "generation_cursor": 0,
                "generation_progress_path": str(_stage_progress_path(output_dir, phase)),
                "generation_completed_counts": completed,
                "generation_rejections": rejections,
                "generation_exhausted": False,
            }
        stage_index += 1
    return {
        "generation_stage_index": len(_STAGES),
        "generation_stage": "",
        "generation_cursor": 0,
        "generation_progress_path": "",
        "generation_completed_counts": completed,
        "generation_rejections": rejections,
        "generation_exhausted": True,
    }


def initialize_generation(request: GenerationRequest) -> dict[str, Any]:
    """Validate a request and initialize the first resumable generation stage."""
    tasks, targets = _validate_tasks(request)
    completed = _all_completed_counts(Path(request["output_directory"]), tasks, targets)
    state = {
        "generation_target_counts": targets,
        "generation_completed_counts": completed,
        "generation_rejections": {},
    }
    return state | _enter_stage(request, tasks, targets, completed, 0, {})


def generation_done(state: dict[str, Any]) -> bool:
    return bool(state.get("generation_exhausted")) and int(state.get("generation_stage_index", 0)) >= len(
        _STAGES
    )


def advance_generation(request: GenerationRequest, state: dict[str, Any]) -> dict[str, Any]:
    """Process one record in the active format stage, then advance stages as needed."""
    tasks, targets = _validate_tasks(request)
    output_dir = Path(request["output_directory"])
    completed = _all_completed_counts(output_dir, tasks, targets)
    rejections = dict(state.get("generation_rejections", {}))
    stage_index = int(state["generation_stage_index"])
    if stage_index >= len(_STAGES):
        return _enter_stage(request, tasks, targets, completed, stage_index, rejections)

    phase, response_format, handler = _STAGES[stage_index]
    stage_tasks = _stage_tasks(tasks, targets, response_format)
    cursor, candidates = _next_candidate_record(
        request,
        stage_tasks,
        targets,
        completed,
        int(state["generation_cursor"]),
        handler,
    )
    if candidates is None:
        record_event(
            request["construction_events_path"], "dataset_exhaustion", run_id=request["run_id"],
            dataset_id=request["dataset_id"], agent=GENERATION_AGENT, operation="generate", component=phase,
            status="complete"
        )
        return _enter_stage(request, tasks, targets, completed, stage_index + 1, rejections)

    accepted: list[dict[str, Any]] = []
    max_tasks = int(request["max_tasks_per_image"])
    temperature = float(request["question_temperature"])
    for candidate in candidates:
        if len(accepted) >= max_tasks:
            break
        item_id = _item_identity(candidate)
        record_event(
            request["construction_events_path"], "generation_attempt", run_id=request["run_id"],
            dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id,
            agent=GENERATION_AGENT, operation="generate", component=phase, status="started"
        )
        with event_context(
            construction_events_path=request["construction_events_path"], run_id=request["run_id"],
            dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id,
            agent=GENERATION_AGENT, operation="generate", component=phase,
            model=get_settings().llm.model,
        ) as generation_call_ids:
            try:
                generated = handler.generate(candidate, request["benchmark_info"], temperature)
            except ModelUnavailable:
                rejections[f"{phase}_generation_model_unavailable"] = rejections.get(
                    f"{phase}_generation_model_unavailable", 0
                ) + 1
                record_event(request["construction_events_path"], "generation_failure", run_id=request["run_id"], dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id, agent=GENERATION_AGENT, operation="generate", component=phase, status="failure", error_type="model_unavailable")
                continue
            except ValueError:
                rejections[f"{phase}_generation_contract"] = rejections.get(
                    f"{phase}_generation_contract", 0
                ) + 1
                record_event(request["construction_events_path"], "generation_failure", run_id=request["run_id"], dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id, agent=GENERATION_AGENT, operation="generate", component=phase, status="failure", error_type="contract")
                continue
        with event_context(
            construction_events_path=request["construction_events_path"], run_id=request["run_id"],
            dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id,
            agent=GENERATION_AGENT, operation="review", component=phase,
            model=get_settings().llm.model,
        ) as review_call_ids:
            try:
                approved, _reason = handler.review(candidate, generated, request["benchmark_info"])
            except ModelUnavailable:
                rejections[f"{phase}_review_model_unavailable"] = rejections.get(
                    f"{phase}_review_model_unavailable", 0
                ) + 1
                record_event(request["construction_events_path"], "review_failure", run_id=request["run_id"], dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id, agent=GENERATION_AGENT, operation="review", component=phase, status="failure", error_type="model_unavailable")
                continue
        if approved:
            item = _build_item(candidate, generated, handler, request["dataset_id"])
            provenance = {
                "item_id": item["id"], "run_id": request["run_id"], "dataset_id": request["dataset_id"],
                "task_id": candidate.task_id, "image_id": item["image_id"], "source_row": candidate.source_row,
                "supporting_fields": candidate.task["supporting_fields"],
                "generation_call_ids": generation_call_ids, "review_call_ids": review_call_ids,
                "generation_attempt_count": len(generation_call_ids), "review_attempt_count": len(review_call_ids),
                "blueprint_hash": request["blueprint_sha256"],
            }
            accepted.append({"item": item, "provenance": provenance})
            record_event(request["construction_events_path"], "review_accept", run_id=request["run_id"], dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id, agent=GENERATION_AGENT, operation="review", component=phase, status="success")
        else:
            rejections[f"{phase}_review_rejected"] = rejections.get(
                f"{phase}_review_rejected", 0
            ) + 1
            record_event(request["construction_events_path"], "review_reject", run_id=request["run_id"], dataset_id=request["dataset_id"], task_id=candidate.task_id, item_id=item_id, agent=GENERATION_AGENT, operation="review", component=phase, status="rejected")

    progress_path = _stage_progress_path(output_dir, phase)
    if accepted:
        _merge_progress(progress_path, accepted, {task_id: index for index, task_id in enumerate(stage_tasks)})
        completed = _all_completed_counts(output_dir, tasks, targets)
    if all(completed[task_id] >= targets[task_id] for task_id in stage_tasks):
        return _enter_stage(request, tasks, targets, completed, stage_index + 1, rejections)
    return {
        "generation_stage_index": stage_index,
        "generation_stage": phase,
        "generation_cursor": cursor,
        "generation_progress_path": str(progress_path),
        "generation_completed_counts": completed,
        "generation_rejections": rejections,
        "generation_exhausted": False,
    }


def finalize_generation(request: GenerationRequest, state: dict[str, Any]) -> GenerationResult:
    """Atomically merge staged progress into raw JSONL and remove temporary files."""
    tasks, targets = _validate_tasks(request)
    output_dir = Path(request["output_directory"])
    kept: list[dict[str, Any]] = []
    counts = {task_id: 0 for task_id in tasks}
    progress_paths: list[Path] = []
    for phase, response_format, _handler in _STAGES:
        progress_path = _stage_progress_path(output_dir, phase)
        progress_paths.append(progress_path)
        stage_tasks = _stage_tasks(tasks, targets, response_format)
        for entry in _read_jsonl(progress_path):
            item = _progress_item(entry)
            task_id = str(item.get("task_id"))
            if task_id in stage_tasks and counts[task_id] < targets[task_id]:
                kept.append(item)
                counts[task_id] += 1
    output = output_dir / f"benchmark.{request['dataset_id']}.raw.jsonl"
    _write_jsonl(output, kept)
    provenance_path = output_dir / "item_provenance.jsonl"
    provenance = [
        entry["provenance"]
        for phase, _response_format, _handler in _STAGES
        for entry in _read_jsonl(_stage_progress_path(output_dir, phase))
        if isinstance(entry.get("provenance"), dict)
    ]
    _write_jsonl(provenance_path, provenance)
    for progress_path in progress_paths:
        if progress_path.exists():
            progress_path.unlink()
    return GenerationResult(str(output), len(kept), counts, str(provenance_path))


def generate_raw_benchmark(request: GenerationRequest) -> GenerationResult:
    """Run the resumable three-stage generator synchronously for direct callers."""
    state = initialize_generation(request)
    while not generation_done(state):
        state.update(advance_generation(request, state))
    return finalize_generation(request, state)
