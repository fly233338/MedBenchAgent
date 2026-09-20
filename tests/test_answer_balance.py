import copy
from collections import Counter

import pytest

from medbenchagent.audit.answer_balance import balance_items


def _item(index: int, answer_value: str) -> dict:
    options = ["benign", "malignant", "indeterminate"]
    return {
        "id": str(index), "task_id": "pathology", "image_id": str(index), "image_path": str(index),
        "question": "What is the status?", "options": options, "answer": chr(ord("A") + options.index(answer_value)),
    }


def test_balance_is_deterministic_and_preserves_answers_and_input() -> None:
    original = [_item(index, "benign" if index % 2 else "malignant") for index in range(7)]
    before = copy.deepcopy(original)
    balanced = balance_items(original, seed=11)
    assert balanced == balance_items(original, seed=11)
    assert original == before
    counts = Counter(item["answer"] for item in balanced)
    all_counts = [counts[letter] for letter in ("A", "B", "C")]
    assert max(all_counts) - min(all_counts) <= 1
    for source, item in zip(original, balanced, strict=True):
        assert item["id"] == source["id"]
        assert item["options"][ord(item["answer"]) - ord("A")] == source["options"][ord(source["answer"]) - ord("A")]


def test_balance_rejects_invalid_option_sets() -> None:
    items = [_item(1, "benign"), {**_item(2, "malignant"), "options": ["benign", "malignant", "other"]}]
    with pytest.raises(ValueError, match="same option collection"):
        balance_items(items)



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

    assert balance_items(records, seed=5) == records
