import json
from pathlib import Path

import pytest

import medbenchagent.generation as generation
from medbenchagent.dataset import prepare_annotation_dataset


def _record(index: int, value: str) -> dict:
    return {
        "image_id": f"img-{index}", "image_path": f"images/{index}.png", "patient_id": None,
        "case_id": None, "study_id": None, "annotation": {"status": value},
    }


def _request(tmp_path: Path, values: list[str], target: int) -> dict:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text("".join(json.dumps(_record(index, value)) + "\n" for index, value in enumerate(values)), encoding="utf-8")
    _profile, manifest = prepare_annotation_dataset(annotation_path, tmp_path / "cache")
    return {
        "benchmark_info": {}, "dataset_access_manifest": manifest, "approved_task_plans": [],
        "question_designs": [{"task_id": "status", "question_templates": ["status"], "citations": []}],
        "closed_tasks": [{"task_id": "status", "task_type": "close_vqa", "description": "Status", "answer_field": "annotation.status", "answer_value_mapping": {}, "options": ["benign", "malignant"]}],
        "target_sample_count": target, "seed": 0, "balance_answer_positions": True, "output_directory": str(tmp_path / "run"),
    }


def test_generation_keeps_options_and_retries_then_uses_next_candidate(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path, ["benign", "malignant"], 1)
    calls = 0

    def fake_complete_json(system: str, _prompt: str) -> dict:
        nonlocal calls
        if system.startswith("You write"):
            calls += 1
            return {} if calls <= 3 else {"question": "What is the image-associated status?"}
        return {"approved": True}

    monkeypatch.setattr(generation, "complete_json", fake_complete_json)
    result = generation.generate_raw_benchmark(request)
    item = json.loads(Path(result.raw_output_path).read_text(encoding="utf-8"))

    assert result.item_count == 1
    assert item["image_ids"] == ["img-1"]
    assert item["options"] == ["benign", "malignant"]
    assert item["answer"] == "B"
    assert item["answer_value"] == "malignant"


def test_generation_fails_before_output_when_candidates_are_insufficient(tmp_path: Path) -> None:
    request = _request(tmp_path, ["benign", "malignant"], 3)

    with pytest.raises(ValueError, match="Insufficient valid candidates"):
        generation.generate_raw_benchmark(request)
    assert not (Path(request["output_directory"]) / "benchmark_items.raw.jsonl").exists()
