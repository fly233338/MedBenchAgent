import json
import pytest
from medbenchagent.agents.task_planning import allocate, finish


def test_dataset_switch_discards_old_plan(tmp_path):
    from medbenchagent.agents.task_planning import select_dataset

    annotation = tmp_path / "annotation.jsonl"
    annotation.touch()
    context = {
        "benchmark_info": {"sample_count": 1},
        "operation_directory": str(tmp_path / "next"),
        "working_blueprint_path": "old.json",
        "plan_binding": ["old.jsonl", "1"],
    }
    outcome = select_dataset(context, annotation_path=str(annotation), dataset_id="new")
    blueprint = json.loads(
        open(outcome["updates"]["working_blueprint_path"], encoding="utf-8").read()
    )
    assert blueprint["dataset"] == {"dataset_id": "new"}
    assert "task_plan" not in blueprint
    assert outcome["updates"]["plan_binding"] is None


def test_finish_rejects_stale_dataset_binding(tmp_path):
    path = tmp_path / "blueprint.json"
    path.write_text(json.dumps({"dataset": {"dataset_id": "set"}, "task_plan": {}, "tasks": {}}))
    with pytest.raises(ValueError, match="different dataset revision"):
        finish(
            {
                "working_blueprint_path": str(path),
                "selected_dataset_id": "set",
                "dataset_profile": {"total_images": 1},
                "dataset_access_manifest": {"revision": "2"},
                "annotation_path": "new.jsonl",
                "plan_binding": ["old.jsonl", "1"],
            },
        )


def test_allocation_tool_preserves_task_design(tmp_path):
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "view": {
                        "category": "image_acquisition_and_basic_information",
                        "description": "Recognize view",
                        "generation_logic": "Recognize",
                    }
                }
            }
        )
    )
    context = {
        "working_blueprint_path": str(path),
        "operation_directory": str(tmp_path / "op"),
        "benchmark_info": {"sample_count": 4},
    }
    output = allocate(context)
    blueprint = json.loads(
        open(output["updates"]["working_blueprint_path"], encoding="utf-8").read()
    )
    assert blueprint["tasks"]["view"]["sample_count"] == 4
    assert blueprint["tasks"]["view"]["generation_logic"] == "Recognize"


def test_planning_finish_requires_complete_design(tmp_path):
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "dataset": {"dataset_id": "set"},
                "task_plan": {},
                "tasks": {"view": {"sample_count": 1}},
            }
        )
    )
    with pytest.raises(ValueError, match="medical design"):
        finish(
            {
                "working_blueprint_path": str(path),
                "benchmark_info": {"sample_count": 1},
                "selected_dataset_id": "set",
                "dataset_profile": {"total_images": 1},
                "dataset_access_manifest": {"annotation_path": "annotation.jsonl", "revision": "1"},
                "annotation_path": "annotation.jsonl",
                "plan_binding": ["annotation.jsonl", "1"],
            },
        )
