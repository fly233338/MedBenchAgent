"""Deterministically balance correct-option positions in closed benchmark items."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rng(seed: int, task_id: str) -> random.Random:
    value = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(value[:8], "big"))


def _validate_group(task_id: str, items: list[dict[str, Any]]) -> tuple[str, ...]:
    first_options: tuple[str, ...] | None = None
    for item in items:
        options = item.get("options")
        answer_value = item.get("answer_value")
        if not isinstance(options, list) or not options or not all(isinstance(option, str) for option in options):
            raise ValueError(f"{task_id}: options must be a non-empty list of strings.")
        option_tuple = tuple(options)
        if len(set(option_tuple)) != len(option_tuple):
            raise ValueError(f"{task_id}: options must be unique.")
        if first_options is None:
            first_options = option_tuple
        elif option_tuple != first_options:
            raise ValueError(f"{task_id}: every item must use the same option collection.")
        if not isinstance(answer_value, str) or options.count(answer_value) != 1:
            raise ValueError(f"{task_id}: answer_value must occur exactly once in options.")
        answer = item.get("answer")
        if not isinstance(answer, str) or len(answer) != 1 or not answer.isalpha():
            raise ValueError(f"{task_id}: answer must be a single option letter.")
        answer_index = ord(answer.upper()) - ord("A")
        if answer_index < 0 or answer_index >= len(options) or options[answer_index] != answer_value:
            raise ValueError(f"{task_id}: answer must point to answer_value.")
    assert first_options is not None
    return first_options


def balance_items(items: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Balance letter-answer positions and pass through other response formats."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every item must have a non-empty task_id.")
        if item.get("response_format") == "letter":
            grouped[task_id].append(item)

    balanced = [dict(item) for item in items]
    copies_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in balanced:
        if item.get("response_format") == "letter":
            copies_by_task[item["task_id"]].append(item)

    for task_id, task_items in grouped.items():
        options = _validate_group(task_id, task_items)
        generator = _rng(seed, task_id)
        count, option_count = len(task_items), len(options)
        positions = list(range(option_count)) * (count // option_count)
        remainder = list(range(option_count))
        generator.shuffle(remainder)
        positions.extend(remainder[: count % option_count])
        generator.shuffle(positions)
        for item, position in zip(copies_by_task[task_id], positions, strict=True):
            answer_value = item["answer_value"]
            wrong_options = [option for option in options if option != answer_value]
            generator.shuffle(wrong_options)
            reordered = wrong_options[:]
            reordered.insert(position, answer_value)
            item["options"] = reordered
            item["answer"] = chr(ord("A") + position)
    return balanced


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number} is not valid JSON.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} must be a JSON object.")
            records.append(record)
    return records


def balance_jsonl(input_path: str | Path, output_path: str | Path, seed: int = 0) -> str:
    """Balance a JSONL copy atomically without modifying the input file."""
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError("Input and output paths must differ; raw benchmark files are immutable.")
    if not source.is_file():
        raise ValueError(f"Input JSONL does not exist: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = balance_items(_read_jsonl(source), seed)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(output)
