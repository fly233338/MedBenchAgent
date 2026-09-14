"""Shared resumable orchestration for staged benchmark generation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.planning.protocol import Protocol, resolve_answer
from medbenchagent.generation.contracts import GenerationTask, GenerationRequest
from medbenchagent.generation.templates import render
from medbenchagent.contracts import (
    BenchmarkItem,
    GenerationResult,
)
from medbenchagent.runtime.construction import GENERATION_AGENT, record_event
from medbenchagent.generation import free_text, grounding, vqa
from medbenchagent.generation.context import (
    has_meaningful_value,
    is_supporting_field_path,
    project_supporting_record,
)
from medbenchagent.metrics import is_metric_supported
from medbenchagent.settings import get_settings

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
    supporting_record: dict[str, Any]
    task: GenerationTask
    question_template: str
    expected_answer: str | None = None
    expected_evidence: dict[str, object] | None = None


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
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in raw_counts
    ):
        raise ValueError("Every Blueprint task must have a non-negative integer sample_count.")
    counts = {task_id: int(task["sample_count"]) for task_id, task in tasks.items()}
    if sum(counts.values()) != target:
        raise ValueError("Task sample counts must sum to target_sample_count.")
    return counts


def _validate_tasks(request: GenerationRequest) -> tuple[dict[str, GenerationTask], dict[str, int]]:
    dataset_id = request.get("dataset_id")
    if not isinstance(dataset_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", dataset_id):
        raise ValueError("dataset_id must be a stable lowercase slug.")
    tasks = request["tasks"]
    targets = _task_targets(request)
    for task_id, task in tasks.items():
        if targets[task_id] == 0:
            continue
        Protocol.model_validate(task["protocol"])
        response_format = task["protocol"].get("response_format")
        if response_format not in {"letter", "bbox", "free_text"}:
            raise ValueError(f"{task_id}: unsupported response_format {response_format}.")
        if not is_metric_supported(response_format, task["protocol"].get("metric")):
            raise ValueError(
                f"{task_id}: final Blueprint task must contain a metric compatible with "
                f"response format {response_format}."
            )
        supporting = task["protocol"]["data_binding"]["supporting_fields"]
        if (
            not isinstance(supporting, list)
            or not supporting
            or not all(is_supporting_field_path(field) for field in supporting)
        ):
            raise ValueError(f"{task_id}: supporting_fields must contain v2 entity field paths.")
        templates = task.get("question_templates")
        if (
            not isinstance(templates, list)
            or not templates
            or not all(isinstance(template, str) and template.strip() for template in templates)
            or len(set(templates)) != len(templates)
        ):
            raise ValueError(
                f"{task_id}: question_templates must contain unique non-empty strings."
            )
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
    return json.dumps(
        [candidate.task_id, candidate.record["image"]["image_id"]], separators=(",", ":")
    )


def _build_item(
    candidate: _Candidate, generated: dict[str, Any], handler: Any, dataset_id: str
) -> BenchmarkItem:
    options, answer = handler.encode_answer(candidate, generated)
    return {
        "id": _item_identity(candidate),
        "task_id": candidate.task_id,
        "dataset": dataset_id,
        "image_id": candidate.record["image"]["image_id"],
        "image_path": candidate.record["image"]["image_path"],
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
                destination.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _progress_item(entry: dict[str, Any]) -> dict[str, Any]:
    return entry


def _progress_counts(items: list[dict[str, Any]], task_ids: list[str]) -> dict[str, int]:
    return {
        task_id: sum(_progress_item(item).get("task_id") == task_id for item in items)
        for task_id in task_ids
    }


def _merge_progress(
    path: Path, additions: list[dict[str, Any]], task_order: dict[str, int]
) -> list[dict[str, Any]]:
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
    record = json.loads(row[5]) if isinstance(row[5], str) else row[5]
    if not isinstance(record, dict):
        raise ValueError("Dataset record must be a JSON object.")
    return {**record, "_source_row": int(row[6])}


def _shuffle_candidates(candidates: list[_Candidate], seed: int, image_id: str) -> None:
    random.Random(seed + int.from_bytes(image_id.encode("utf-8"), "big")).shuffle(candidates)


def _select_question_template(
    task: GenerationTask, seed: int, task_id: str, image_id: str, source_row: int
) -> str:
    templates = task["question_templates"]
    identity = json.dumps([task_id, image_id, source_row]).encode("utf-8")
    return random.Random(seed + int.from_bytes(identity, "big")).choice(templates)


def _all_supporting_fields_present(record: dict[str, Any], supporting_fields: list[str]) -> bool:
    return all(
        has_meaningful_value(project_supporting_record(record, [field]))
        for field in supporting_fields
    )


def _mapped_answer(
    record: dict[str, Any], task: GenerationTask
) -> tuple[str, dict[str, object]] | None:
    resolved = resolve_answer(record, task["protocol"])
    return (resolved["answer"], resolved["evidence"]) if resolved else None


def _next_candidate_record(
    request: GenerationRequest,
    tasks: dict[str, GenerationTask],
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
                "SELECT image_id, image_path, patient_id, case_id, study_id, record, _source_row "
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
                    if not _all_supporting_fields_present(record, task["protocol"]["data_binding"]["supporting_fields"]):
                        continue
                    if resolve_answer(record, task["protocol"]) is None:
                        continue
                    projected = project_supporting_record(record, task["protocol"]["data_binding"]["supporting_fields"])
                    image_id = str(record["image"]["image_id"])
                    mapped = (
                        _mapped_answer(record, task)
                        if task["protocol"]["response_format"] == "letter"
                        else None
                    )
                    if task["protocol"]["response_format"] == "letter" and mapped is None:
                        continue
                    candidate = _Candidate(
                        task_id,
                        record["_source_row"],
                        record,
                        projected,
                        task,
                        _select_question_template(
                            task, seed, task_id, image_id, record["_source_row"]
                        ),
                        mapped[0] if mapped else None,
                        mapped[1] if mapped else None,
                    )
                    if has_meaningful_value(projected) and handler.supports(candidate):
                        candidates.append(candidate)
                if candidates:
                    _shuffle_candidates(candidates, seed, image_id)
                    return cursor, candidates
            if len(rows) < fetch_batch_size:
                return cursor, None
    finally:
        connection.close()


def _stage_progress_path(output_dir: Path, phase: str) -> Path:
    return output_dir / f"benchmark_items.{phase}.progress.jsonl"


def _stage_tasks(
    tasks: dict[str, GenerationTask], targets: dict[str, int], response_format: str
) -> dict[str, GenerationTask]:
    return {
        task_id: task
        for task_id, task in tasks.items()
        if task["protocol"]["response_format"] == response_format and targets[task_id] > 0
    }


def _all_completed_counts(
    output_dir: Path, tasks: dict[str, GenerationTask], targets: dict[str, int]
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
    tasks: dict[str, GenerationTask],
    targets: dict[str, int],
    completed: dict[str, int],
    stage_index: int,
    rejections: dict[str, int],
) -> dict[str, Any]:
    output_dir = Path(request["output_directory"])
    while stage_index < len(_STAGES):
        phase, response_format, _handler = _STAGES[stage_index]
        stage_tasks = _stage_tasks(tasks, targets, response_format)
        if stage_tasks and not all(
            completed[task_id] >= targets[task_id] for task_id in stage_tasks
        ):
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
    raw = Path(request["output_directory"]) / f"benchmark.{request['dataset_id']}.raw.jsonl"
    if raw.is_file():
        items = _read_jsonl(raw)
        completed = _progress_counts(items, list(tasks))
        return {
            "generation_target_counts": targets,
            "generation_completed_counts": completed,
            "generation_rejections": {},
            "generation_stage_index": len(_STAGES),
            "generation_stage": "",
            "generation_cursor": 0,
            "generation_exhausted": True,
        }
    completed = _all_completed_counts(Path(request["output_directory"]), tasks, targets)
    state = {
        "generation_target_counts": targets,
        "generation_completed_counts": completed,
        "generation_rejections": {},
    }
    return state | _enter_stage(request, tasks, targets, completed, 0, {})


def generation_done(state: dict[str, Any]) -> bool:
    return bool(state.get("generation_exhausted")) and int(
        state.get("generation_stage_index", 0)
    ) >= len(_STAGES)


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
            request["construction_events_path"],
            "dataset_exhaustion",
            run_id=request["run_id"],
            dataset_id=request["dataset_id"],
            agent=GENERATION_AGENT,
            operation="generate",
            component=phase,
            status="complete",
        )
        return _enter_stage(request, tasks, targets, completed, stage_index + 1, rejections)

    accepted: list[dict[str, Any]] = []
    max_tasks = int(request["max_tasks_per_image"])
    existing = _read_jsonl(_stage_progress_path(output_dir, phase))
    image_id = candidates[0].record["image"]["image_id"] if candidates else None
    existing_count = sum(item["image_id"] == image_id for item in existing)
    existing_ids = {item["id"] for item in existing}
    for candidate in candidates:
        if len(accepted) + existing_count >= max_tasks:
            break
        item_id = _item_identity(candidate)
        if item_id in existing_ids:
            continue
        record_event(
            request["construction_events_path"],
            "generation_attempt",
            run_id=request["run_id"],
            dataset_id=request["dataset_id"],
            task_id=candidate.task_id,
            item_id=item_id,
            agent=GENERATION_AGENT,
            operation="generate",
            component=phase,
            status="started",
        )
        try:
            generated = handler._validate(candidate, {"question": render(candidate)})
            item = _build_item(candidate, generated, handler, request["dataset_id"])
            accepted.append(item)
        except ValueError as exc:
            rejections[f"{phase}_contract"] = rejections.get(f"{phase}_contract", 0) + 1
            record_event(
                request["construction_events_path"],
                "generation_reject",
                run_id=request["run_id"],
                task_id=candidate.task_id,
                item_id=item_id,
                agent=GENERATION_AGENT,
                operation="instantiate",
                status="rejected",
                reason=str(exc),
            )

    progress_path = _stage_progress_path(output_dir, phase)
    if accepted:
        _merge_progress(
            progress_path, accepted, {task_id: index for index, task_id in enumerate(stage_tasks)}
        )
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


def finalize_generation(request: GenerationRequest) -> GenerationResult:
    """Atomically merge staged progress into raw JSONL and remove temporary files."""
    tasks, targets = _validate_tasks(request)
    output_dir = Path(request["output_directory"])
    output = output_dir / f"benchmark.{request['dataset_id']}.raw.jsonl"
    if output.is_file():
        items = _read_jsonl(output)
        return GenerationResult(str(output), len(items), _progress_counts(items, list(tasks)))
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
    for progress_path in progress_paths:
        if progress_path.exists():
            progress_path.unlink()
    return GenerationResult(str(output), len(kept), counts)


def generate_raw_benchmark(request: GenerationRequest) -> GenerationResult:
    """Run the resumable three-stage generator synchronously for direct callers."""
    state = initialize_generation(request)
    while not generation_done(state):
        state.update(advance_generation(request, state))
    return finalize_generation(request)
