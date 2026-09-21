"""Deterministically balance correct-option positions in closed benchmark items."""

from __future__ import annotations

import random
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


def balance_items(items, counts, seed=0):
    """Draw balanced positions without materializing items or a per-item position list."""
    groups = {}
    for original in items:
        if not isinstance(original.get("task_id"), str) or not original["task_id"]:
            raise ValueError("Every item must have a non-empty task_id")
        if not isinstance(original.get("options"), list):
            yield original
            continue
        item = dict(original)
        task_id = item["task_id"]
        options = _validate_group(task_id, [item])
        if task_id not in groups:
            generator = _rng(seed, task_id)
            remaining = [counts[task_id] // len(options)] * len(options)
            order = list(range(len(options)))
            generator.shuffle(order)
            for position in order[:counts[task_id] % len(options)]:
                remaining[position] += 1
            groups[task_id] = options, generator, remaining
        expected, generator, remaining = groups[task_id]
        if options != expected:
            raise ValueError(f"{task_id}: every item must use the same option collection")
        if not sum(remaining):
            raise ValueError("Candidate counts changed during balancing")
        draw = generator.randrange(sum(remaining))
        for position, count in enumerate(remaining):
            if draw < count:
                break
            draw -= count
        remaining[position] -= 1
        answer = options[ord(item["answer"].upper()) - 65]
        wrong = [value for value in options if value != answer]
        generator.shuffle(wrong)
        wrong.insert(position, answer)
        item.update(options=wrong, answer=chr(65 + position))
        yield item
    if set(groups) != set(counts) or any(sum(group[2]) for group in groups.values()):
        raise ValueError("Candidate counts changed during balancing")
