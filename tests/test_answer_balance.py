import json
from collections import Counter
from pathlib import Path

import pytest

from medbenchagent.answer_balance import balance_jsonl


def _item(index: int, answer_value: str) -> dict:
    options = ["benign", "malignant", "indeterminate"]
    return {
        "id": str(index), "task_id": "pathology", "image_id": str(index), "image_path": str(index),
        "question": "What is the status?", "options": options, "answer": chr(ord("A") + options.index(answer_value)),
    }


def _write(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_balance_writes_deterministic_copy_and_preserves_raw(tmp_path: Path) -> None:
    raw = tmp_path / "benchmark_items.raw.jsonl"
    original = [_item(index, "benign" if index % 2 else "malignant") for index in range(7)]
    _write(raw, original)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    balance_jsonl(raw, first, seed=11)
    balance_jsonl(raw, second, seed=11)

    assert first.read_bytes() == second.read_bytes()
    assert [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()] == original
    balanced = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    counts = Counter(item["answer"] for item in balanced)
    all_counts = [counts[letter] for letter in ("A", "B", "C")]
    assert max(all_counts) - min(all_counts) <= 1
    for before, after in zip(original, balanced, strict=True):
        assert after["id"] == before["id"]
        assert after["options"][ord(after["answer"]) - ord("A")] == before["options"][ord(before["answer"]) - ord("A")]

    different_seed = tmp_path / "different-seed.jsonl"
    balance_jsonl(raw, different_seed, seed=12)
    for item in (json.loads(line) for line in different_seed.read_text(encoding="utf-8").splitlines()):
        assert item["options"][ord(item["answer"]) - ord("A")] in item["options"]


def test_balance_rejects_invalid_option_sets_and_same_path(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    _write(raw, [_item(1, "benign"), {**_item(2, "malignant"), "options": ["benign", "malignant", "other"]}])

    with pytest.raises(ValueError, match="same option collection"):
        balance_jsonl(raw, tmp_path / "balanced.jsonl")
    with pytest.raises(ValueError, match="must differ"):
        balance_jsonl(raw, raw)


def test_balance_passes_through_bbox_and_free_text_items() -> None:
    records = [
        {
            "id": "bbox", "task_id": "detection", "image_id": "img", "image_path": "img.png",
            "question": "Locate the mass.", "options": None, "answer": [1.0, 2.0, 3.0, 4.0],
        },
        {
            "id": "report", "task_id": "report_generation", "image_id": "img", "image_path": "img.png",
            "question": "Generate a report.", "options": None, "answer": "Normal study.",
        },
    ]

    from medbenchagent.answer_balance import balance_items

    assert balance_items(records, seed=5) == records
