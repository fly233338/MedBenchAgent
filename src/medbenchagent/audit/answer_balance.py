"""Deterministically balance correct-option positions in closed benchmark items."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def _rng(seed: int, task_id: str) -> random.Random:
    return random.Random(seed + int.from_bytes(task_id.encode("utf-8"), "big"))


def _validate_group(task_id: str, items: list[dict[str, Any]]) -> tuple[str, ...]:
    first_options: tuple[str, ...] | None = None
    for item in items:
        options = item.get("options")
        if not isinstance(options, list) or not options or not all(isinstance(option, str) for option in options):
            raise ValueError(f"{task_id}: options must be a non-empty list of strings.")
        option_tuple = tuple(options)
        if len(set(option_tuple)) != len(option_tuple):
            raise ValueError(f"{task_id}: options must be unique.")
        if first_options is None:
            first_options = option_tuple
        elif option_tuple != first_options:
            raise ValueError(f"{task_id}: every item must use the same option collection.")
        answer = item.get("answer")
        if not isinstance(answer, str) or len(answer) != 1 or not answer.isalpha():
            raise ValueError(f"{task_id}: answer must be a single option letter.")
        answer_index = ord(answer.upper()) - ord("A")
        if answer_index < 0 or answer_index >= len(options):
            raise ValueError(f"{task_id}: answer must point to an option.")
    assert first_options is not None
    return first_options


def balance_items(items: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Balance letter-answer positions and pass through other response formats."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("Every item must have a non-empty task_id.")
        if isinstance(item.get("options"), list):
            grouped[task_id].append(item)

    balanced = [dict(item) for item in items]
    copies_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in balanced:
        if isinstance(item.get("options"), list):
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
            answer_value = item["options"][ord(item["answer"].upper()) - ord("A")]
            wrong_options = [option for option in options if option != answer_value]
            generator.shuffle(wrong_options)
            reordered = wrong_options[:]
            reordered.insert(position, answer_value)
            item["options"] = reordered
            item["answer"] = chr(ord("A") + position)
    return balanced
