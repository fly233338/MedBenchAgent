"""Mixed benchmark generation from annotations and approved Question Designs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from medbenchagent.contracts import BenchmarkItem, GenerationRequest, GenerationResult, TaskGenerationSpec
from medbenchagent.llm import ModelUnavailable, complete_json

MAX_ATTEMPTS_PER_CANDIDATE = 3


@dataclass(frozen=True)
class _Candidate:
    task_id: str
    source_row: int
    instance_index: int | None
    record: dict[str, Any]
    answer: str | list[float]
    answer_value: str | None
    bindings: dict[str, str]
    spec: TaskGenerationSpec

    @property
    def key(self) -> tuple[str, int, int | None]:
        return self.task_id, self.source_row, self.instance_index


def _path_values(root: object, segments: list[str]) -> list[object]:
    values = [root]
    for segment in segments:
        is_array = segment.endswith("[*]")
        key = segment.removesuffix("[*]")
        nested_values: list[object] = []
        for value in values:
            if not isinstance(value, dict) or key not in value:
                continue
            nested = value[key]
            if is_array:
                if isinstance(nested, list):
                    nested_values.extend(nested)
            else:
                nested_values.append(nested)
        values = nested_values
    return values


def _annotation_values(annotation: dict[str, Any], path: str) -> list[object]:
    if not path.startswith("annotation."):
        raise ValueError(f"Annotation field must start with annotation.: {path}")
    return _path_values(annotation, path.split(".")[1:])


def _instances(annotation: dict[str, Any], instance_path: str) -> list[dict[str, Any]]:
    if not instance_path.startswith("annotation.") or not instance_path.endswith("[*]"):
        raise ValueError(f"instance_path must identify an annotation array: {instance_path}")
    return [value for value in _annotation_values(annotation, instance_path) if isinstance(value, dict)]


def _bound_values(
    annotation: dict[str, Any], field: str, instance_path: str | None, instance: dict[str, Any] | None
) -> list[object]:
    if instance_path is not None and field.startswith(instance_path + "."):
        if instance is None:
            return []
        relative = field[len(instance_path) + 1 :]
        return _path_values(instance, relative.split("."))
    return _annotation_values(annotation, field)


def _one_scalar(values: list[object]) -> object | None:
    non_empty = [value for value in values if value not in (None, "")]
    if len(non_empty) != 1 or isinstance(non_empty[0], (dict, list)):
        return None
    return non_empty[0]


def _serialized_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_task_specs(request: GenerationRequest) -> list[TaskGenerationSpec]:
    specs = request.get("task_specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("No retained task is available for generation.")
    task_ids: set[str] = set()
    for spec in specs:
        task_id = spec.get("task_id")
        response_format = spec.get("response_format")
        templates = spec.get("question_templates")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            raise ValueError("Task generation specs must have unique non-empty task_id values.")
        task_ids.add(task_id)
        if not isinstance(templates, list) or not templates or not all(
            isinstance(template, str) and template for template in templates
        ):
            raise ValueError(f"{task_id}: question_templates must be non-empty strings.")
        if response_format == "letter":
            options = spec.get("options")
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 26
                or not all(isinstance(option, str) for option in options)
                or len(set(options)) != len(options)
                or not isinstance(spec.get("answer_field"), str)
            ):
                raise ValueError(f"{task_id}: letter tasks need a valid answer_field and fixed options.")
        elif response_format == "bbox":
            if (
                not isinstance(spec.get("instance_path"), str)
                or not isinstance(spec.get("target_text_field"), str)
                or not isinstance(spec.get("bbox_field"), str)
                or spec.get("bbox_format") != "xyxy"
            ):
                raise ValueError(f"{task_id}: bbox tasks need aligned target and xyxy fields.")
        elif response_format == "free_text":
            if not isinstance(spec.get("answer_field"), str):
                raise ValueError(f"{task_id}: free_text tasks need an answer_field.")
        else:
            raise ValueError(f"{task_id}: unsupported response_format {response_format}.")
    return specs


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


def _template_bindings(
    spec: TaskGenerationSpec,
    annotation: dict[str, Any],
    instance: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    instance_path = spec.get("instance_path")
    bindings: dict[str, str] = {}
    for name, field in spec.get("placeholder_bindings", {}).items():
        value = _one_scalar(_bound_values(annotation, field, instance_path, instance))
        if value is None:
            return None
        bindings[name] = str(value)
    return bindings


def _closed_candidate(spec: TaskGenerationSpec, record: dict[str, Any]) -> _Candidate | None:
    field = spec["answer_field"]
    value = _one_scalar(_annotation_values(record["annotation"], field))
    if value is None:
        return None
    mapping = spec.get("answer_value_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"{spec['task_id']}: answer_value_mapping must be an object.")
    answer_value = mapping.get(_serialized_value(value), mapping.get(str(value), value))
    if not isinstance(answer_value, str) or answer_value not in spec["options"]:
        return None
    return _Candidate(
        spec["task_id"], record["_source_row"], None, record, answer_value, answer_value, {}, spec
    )


def _free_text_candidates(spec: TaskGenerationSpec, record: dict[str, Any]) -> list[_Candidate]:
    instance_path = spec.get("instance_path")
    instances: list[dict[str, Any] | None] = (
        list(_instances(record["annotation"], instance_path)) if instance_path else [None]
    )
    candidates: list[_Candidate] = []
    for index, instance in enumerate(instances):
        value = _one_scalar(
            _bound_values(record["annotation"], spec["answer_field"], instance_path, instance)
        )
        bindings = _template_bindings(spec, record["annotation"], instance)
        if not isinstance(value, str) or not value.strip() or bindings is None:
            continue
        candidates.append(
            _Candidate(
                spec["task_id"],
                record["_source_row"],
                index if instance_path else None,
                record,
                value.strip(),
                None,
                bindings,
                spec,
            )
        )
    return candidates


def _bbox_value(value: object) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(isinstance(number, (int, float)) and not isinstance(number, bool) for number in value)
    ):
        return None
    bbox = [float(number) for number in value]
    return bbox if bbox[2] > bbox[0] and bbox[3] > bbox[1] else None


def _bbox_candidates(spec: TaskGenerationSpec, record: dict[str, Any]) -> list[_Candidate]:
    instance_path = spec["instance_path"]
    candidates: list[_Candidate] = []
    for index, instance in enumerate(_instances(record["annotation"], instance_path)):
        target = _one_scalar(
            _bound_values(record["annotation"], spec["target_text_field"], instance_path, instance)
        )
        bbox_values = _bound_values(record["annotation"], spec["bbox_field"], instance_path, instance)
        bbox = _bbox_value(bbox_values[0]) if len(bbox_values) == 1 else _bbox_value(bbox_values)
        bindings = _template_bindings(spec, record["annotation"], instance)
        if not isinstance(target, str) or not target.strip() or bbox is None or bindings is None:
            continue
        candidates.append(
            _Candidate(
                spec["task_id"], record["_source_row"], index, record, bbox, None, bindings, spec
            )
        )
    return candidates


def _load_candidates(
    request: GenerationRequest, specs: list[TaskGenerationSpec]
) -> dict[str, list[_Candidate]]:
    candidates: dict[str, list[_Candidate]] = defaultdict(list)
    for record in _load_records(request):
        for spec in specs:
            if spec["response_format"] == "letter":
                candidate = _closed_candidate(spec, record)
                if candidate is not None:
                    candidates[spec["task_id"]].append(candidate)
            elif spec["response_format"] == "bbox":
                candidates[spec["task_id"]].extend(_bbox_candidates(spec, record))
            else:
                candidates[spec["task_id"]].extend(_free_text_candidates(spec, record))
    return candidates


def _safe_wording_design(spec: TaskGenerationSpec) -> dict[str, object]:
    return {
        "question_templates": spec["question_templates"],
        "placeholder_bindings": spec.get("placeholder_bindings", {}),
        "validation_rules": spec.get("validation_rules", []),
        "citations": spec.get("citations", []),
    }


def _closed_question(candidate: _Candidate) -> str | None:
    prompt = (
        "Generate one concise closed medical VQA question. Return JSON only as {\"question\":\"...\"}. "
        "Do not state, alter, infer, or mention the answer, answer options, or label.\n\n"
        f"Task: {candidate.task_id}\nDescription: {candidate.spec.get('description', '')}\n"
        f"Question design: {json.dumps(_safe_wording_design(candidate.spec), ensure_ascii=False)}"
    )
    try:
        response = complete_json("You write only the question wording for a benchmark item.", prompt)
    except ModelUnavailable:
        return None
    if set(response) != {"question"} or not isinstance(response["question"], str):
        return None
    question = response["question"].strip()
    return question or None


def _approved(question: str, candidate: _Candidate) -> bool:
    try:
        response = complete_json(
            "Independently review benchmark question wording. Return JSON only as {\"approved\":true|false}.",
            f"Task: {candidate.task_id}\nDescription: {candidate.spec.get('description', '')}\n"
            f"Question design: {json.dumps(_safe_wording_design(candidate.spec), ensure_ascii=False)}\n"
            f"Question: {question}\nApprove only if it is clear, follows the design, and does not expose an answer.",
        )
    except ModelUnavailable:
        return False
    return set(response) == {"approved"} and response["approved"] is True


def _item_identity(candidate: _Candidate) -> str:
    raw = (
        f"{candidate.task_id}:{candidate.record['image_id']}:{candidate.source_row}:"
        f"{candidate.instance_index if candidate.instance_index is not None else ''}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _template_question(candidate: _Candidate, seed: int) -> str:
    templates = candidate.spec["question_templates"]
    digest = hashlib.sha256(f"{seed}:{_item_identity(candidate)}".encode("utf-8")).digest()
    template = templates[int.from_bytes(digest[:8], "big") % len(templates)]
    try:
        question = template.format_map(candidate.bindings).strip()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{candidate.task_id}: cannot instantiate Question Design template.") from exc
    if not question:
        raise ValueError(f"{candidate.task_id}: instantiated question is empty.")
    return question


def _build_item(candidate: _Candidate, question: str) -> BenchmarkItem:
    response_format = candidate.spec["response_format"]
    source_ref: dict[str, object] = {
        "image_id": candidate.record["image_id"],
        "source_row": candidate.source_row,
    }
    if candidate.instance_index is not None:
        source_ref["instance_index"] = candidate.instance_index
    item: BenchmarkItem = {
        "id": _item_identity(candidate),
        "task_id": candidate.task_id,
        "task_type": candidate.spec["task_type"],
        "response_format": response_format,
        "image_ids": [candidate.record["image_id"]],
        "image_paths": [candidate.record["image_path"]],
        "question": question,
        "options": None,
        "answer": candidate.answer,
        "source_refs": [source_ref],
        "medical_sources": list(candidate.spec.get("citations", [])),
    }
    if response_format == "letter":
        options = list(candidate.spec["options"])
        assert candidate.answer_value is not None
        item["options"] = options
        item["answer"] = chr(ord("A") + options.index(candidate.answer_value))
        item["answer_value"] = candidate.answer_value
        source_ref["answer_field"] = candidate.spec["answer_field"]
    elif response_format == "bbox":
        source_ref["target_text_field"] = candidate.spec["target_text_field"]
        source_ref["bbox_field"] = candidate.spec["bbox_field"]
        source_ref["bbox_format"] = "xyxy"
    else:
        source_ref["answer_field"] = candidate.spec["answer_field"]
    return item


def _generate_from_candidates(
    candidates: list[_Candidate],
    needed: int,
    consumed: set[tuple[str, int, int | None]],
    seed: int,
) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for candidate in candidates:
        if candidate.key in consumed:
            continue
        consumed.add(candidate.key)
        if candidate.spec["response_format"] == "letter":
            question = None
            for _attempt in range(MAX_ATTEMPTS_PER_CANDIDATE):
                generated = _closed_question(candidate)
                if generated and _approved(generated, candidate):
                    question = generated
                    break
            if question is None:
                continue
        else:
            question = _template_question(candidate, seed)
        items.append(_build_item(candidate, question))
        if len(items) == needed:
            break
    return items


def generate_raw_benchmark(request: GenerationRequest) -> GenerationResult:
    """Generate exactly target_sample_count mixed benchmark items atomically."""
    target = request["target_sample_count"]
    if not isinstance(target, int) or target <= 0:
        raise ValueError("target_sample_count must be a positive integer.")
    specs = _validate_task_specs(request)
    candidates = _load_candidates(request, specs)
    active_specs = [spec for spec in specs if candidates[spec["task_id"]]]
    if not active_specs or sum(len(candidates[spec["task_id"]]) for spec in active_specs) < target:
        raise ValueError("Insufficient valid candidates to reach target_sample_count.")

    quotas = {spec["task_id"]: target // len(active_specs) for spec in active_specs}
    for spec in active_specs[: target % len(active_specs)]:
        quotas[spec["task_id"]] += 1
    consumed: set[tuple[str, int, int | None]] = set()
    items: list[BenchmarkItem] = []
    seed = request["seed"]
    for spec in active_specs:
        items.extend(
            _generate_from_candidates(
                candidates[spec["task_id"]], quotas[spec["task_id"]], consumed, seed
            )
        )
    shortfall = target - len(items)
    if shortfall:
        for spec in active_specs:
            additions = _generate_from_candidates(
                candidates[spec["task_id"]], shortfall, consumed, seed
            )
            items.extend(additions)
            shortfall -= len(additions)
            if not shortfall:
                break
    if shortfall:
        raise ValueError("Unable to generate target_sample_count validated benchmark items.")

    task_order = {spec["task_id"]: index for index, spec in enumerate(specs)}
    items.sort(
        key=lambda item: (
            task_order[item["task_id"]],
            item["source_refs"][0]["source_row"],
            item["source_refs"][0].get("instance_index", -1),
        )
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
    counts = {
        spec["task_id"]: sum(item["task_id"] == spec["task_id"] for item in items) for spec in specs
    }
    return GenerationResult(str(output), len(items), counts)
