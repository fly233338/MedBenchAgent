import hashlib
import json
from pathlib import Path

import pytest

import medbenchagent.agents.capability as capability_agent
from medbenchagent.construction import construction_report
from medbenchagent.task_screening import TaskPlanReviewRequired


from graph_test_support import _task


def _datasets() -> list[dict]:
    return [
        {
            "id": "vindr-mammo",
            "name": "VinDr-Mammo",
            "description": "Mammography annotations.",
            "annotation_path": "datasets/Mammography/Vin-Dr-normalized/annotation.jsonl",
            "available": True,
        },
        {
            "id": "future",
            "name": "Future Dataset",
            "description": "Not normalized yet.",
            "annotation_path": "datasets/future/annotation.jsonl",
            "available": False,
        },
    ]


def test_dataset_recommendation_and_profile_keep_paths_private(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(capability_agent, "load_dataset_catalog", _datasets)
    monkeypatch.setattr(
        capability_agent,
        "recommend_dataset",
        lambda _benchmark_info, _datasets: {"dataset_id": "vindr-mammo", "reason": "Matches."},
    )

    result = capability_agent._dataset_recommend_node(
        {"benchmark_info": {"modality": "mammography"}}
    )

    assert result["dataset_recommendation"] == {
        "dataset_id": "vindr-mammo",
        "name": "VinDr-Mammo",
        "description": "Mammography annotations.",
        "reason": "Matches.",
    }
    assert "annotation_path" not in result["dataset_recommendation"]
    assert result["dataset_confirmed"] is False

    blueprint_path = tmp_path / "blueprint.json"
    blueprint = {"dataset": {"dataset_id": "vindr-mammo"}, "tasks": {}}
    blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")
    expected_path = tmp_path / "annotation.jsonl"
    monkeypatch.setattr(
        capability_agent,
        "resolve_annotation_path",
        lambda _dataset: expected_path,
    )

    def prepare(path, _cache_root):
        assert path == expected_path
        return {"schema_version": "task-capability-profile.v3", "total_images": 2}, {
            "annotation_path": str(expected_path),
            "duckdb_path": "dataset.duckdb",
            "annotation_sha256": "a" * 64,
            "cache_format_version": "annotation-cache.v6",
        }

    monkeypatch.setattr(capability_agent, "prepare_annotation_dataset", prepare)

    profile_result = capability_agent._dataset_profile_node(
        {"selected_dataset_id": "vindr-mammo", "working_blueprint_path": str(blueprint_path)}
    )

    assert json.loads(blueprint_path.read_text(encoding="utf-8")) == blueprint
    assert profile_result["dataset_profile"] == {
        "schema_version": "task-capability-profile.v3",
        "total_images": 2,
    }
    assert profile_result["dataset_access_manifest"] == {
        "duckdb_path": "dataset.duckdb",
        "annotation_sha256": "a" * 64,
        "cache_format_version": "annotation-cache.v6",
    }
    assert "annotation_path" not in profile_result


def test_dataset_review_confirms_reselects_and_rejects_unavailable_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps({"tasks": {}}), encoding="utf-8")
    state = {
        "working_blueprint_path": str(blueprint_path),
        "dataset_recommendation": {
            "dataset_id": "vindr-mammo",
            "name": "VinDr-Mammo",
            "description": "Mammography annotations.",
            "reason": "Matches.",
        },
    }
    monkeypatch.setattr(capability_agent, "load_dataset_catalog", _datasets)
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "确认")
    monkeypatch.setattr(capability_agent, "complete_json", lambda *_args: {"action": "confirm"})
    confirmed = capability_agent._dataset_review_node(state)
    assert confirmed == {"selected_dataset_id": "vindr-mammo", "dataset_confirmed": True}
    assert json.loads(blueprint_path.read_text(encoding="utf-8"))["dataset"] == {
        "dataset_id": "vindr-mammo"
    }

    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "使用 VinDr")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "select", "dataset_id": "vindr-mammo"},
    )
    reselected = capability_agent._dataset_review_node(state)
    assert reselected["dataset_confirmed"] is False
    assert reselected["selected_dataset_id"] == ""

    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "select", "dataset_id": "future"},
    )
    with pytest.raises(capability_agent.DatasetCatalogError, match="not available"):
        capability_agent._dataset_review_action("使用 Future Dataset", _datasets())


def test_dataset_reselection_records_actual_dataset_change(tmp_path: Path, monkeypatch) -> None:
    datasets = _datasets()
    datasets[1]["available"] = True
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps({"tasks": {}}), encoding="utf-8")
    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    state = {
        "run_id": "run",
        "construction_events_path": str(events_path),
        "working_blueprint_path": str(blueprint_path),
        "dataset_recommendation": {
            "dataset_id": "vindr-mammo",
            "name": "VinDr-Mammo",
            "description": "Mammography annotations.",
            "reason": "Matches.",
        },
    }
    monkeypatch.setattr(capability_agent, "load_dataset_catalog", lambda: datasets)
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "使用 future")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "select", "dataset_id": "future"},
    )

    capability_agent._dataset_review_node(state)

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["action"] == "revise"
    assert events[1]["modified_object_ids"] == ["future"]
    assert construction_report(events_path)["human_stages"]["dataset"] == {
        "confirmed": 0,
        "revision_rounds": 1,
        "modified_objects": 1,
    }


def test_material_selection_uses_exact_scanned_paths_and_reprompts_on_invalid_choice(
    tmp_path: Path, monkeypatch
) -> None:
    materials = [
        {
            "category": "Standard",
            "file": "Standard/guide.pdf",
        }
    ]
    monkeypatch.setattr(capability_agent, "discover_medical_materials", lambda _root: materials)
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "选择指南")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"selected_files": [materials[0]["file"]]},
    )

    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    state = {"run_id": "run", "construction_events_path": str(events_path)}
    selected = capability_agent._materials_choose_node(state)

    assert selected == {
        "selected_material_files": [materials[0]["file"]],
        "material_selection_error": "",
    }

    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"selected_files": ["missing.pdf"]},
    )
    invalid = capability_agent._materials_choose_node(state)

    assert invalid["selected_material_files"] == []
    assert "Unknown medical PDF selection" in invalid["material_selection_error"]
    interactions = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(event["parse_status"], event["user_input"]) for event in interactions] == [
        ("success", "选择指南"),
        ("failed", "选择指南"),
    ]
    assert interactions[0]["selected_objects"] == 1
    assert construction_report(events_path)["human_stages"]["materials"] == {
        "confirmed": 1,
        "revision_rounds": 0,
        "modified_objects": 0,
        "selected_objects": 1,
    }


def test_sample_counts_are_weighted_prepared_and_written_only_after_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"sample_count": 5},
                "tasks": {
                    "view": _task(category="image_acquisition_and_basic_information"),
                    "diagnosis": _task(category="clinical_assessment_and_diagnosis"),
                },
            }
        ),
        encoding="utf-8",
    )
    state = {
        "working_blueprint_path": str(path),
        "benchmark_info": {"sample_count": 5},
    }

    prepared = capability_agent._sample_count_prepare_node(state)
    assert prepared["draft_sample_counts"] == {"view": 2, "diagnosis": 3}
    assert "sample_count" not in json.loads(path.read_text(encoding="utf-8"))["tasks"]["view"]

    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "确认")
    monkeypatch.setattr(capability_agent, "complete_json", lambda *_args: {"action": "confirm"})
    reviewed = capability_agent._sample_count_review_node(state | prepared)

    assert reviewed["sample_counts_confirmed"] is True
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    assert {task_id: task["sample_count"] for task_id, task in tasks.items()} == {
        "view": 2,
        "diagnosis": 3,
    }


def test_task_plan_revision_records_only_tasks_with_actual_changes(
    tmp_path: Path, monkeypatch
) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps({"tasks": {}}), encoding="utf-8")
    original = {"tasks": {"view": {"decision": "drop"}, "report": {"decision": "retain"}}}
    revised = {"tasks": {"view": {"decision": "retain"}, "report": {"decision": "retain"}}}
    state = {
        "run_id": "run",
        "construction_events_path": str(events_path),
        "working_blueprint_path": str(blueprint_path),
        "dataset_profile": {},
        "draft_task_plan": original,
    }
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "保留 view")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "revise", "task_id": "view", "feedback": "Keep it."},
    )
    monkeypatch.setattr(capability_agent, "revise_task_plan", lambda *_args: revised)
    capability_agent.write_task_plan_report(blueprint_path, original)

    review = capability_agent._task_plan_review_node(state)
    revised_state = capability_agent._task_plan_revise_node(state | review)

    report = Path(revised_state["task_screening_report_path"])
    assert report == tmp_path / "task_plans" / "v2.json"
    assert json.loads(report.read_text(encoding="utf-8")) == revised
    assert json.loads((tmp_path / "task_plans" / "v1.json").read_text(encoding="utf-8")) == original

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[1]["modified_object_ids"] == ["view"]
    assert events[1]["artifact_path"] == "task_plans/v2.json"


def test_task_plan_draft_is_written_before_human_review(tmp_path: Path, monkeypatch) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    blueprint_path.write_text(json.dumps({"tasks": {"view": _task()}}), encoding="utf-8")
    plan = {
        "tasks": {
            "view": {
                "decision": "retain",
                "supporting_fields": ["image.annotation.view"],
                "metric": "accuracy",
                "reason": "Available.",
            }
        }
    }
    monkeypatch.setattr(capability_agent, "generate_task_plan", lambda *_args: plan)

    result = capability_agent._task_plan_generate_node(
        {
            "run_id": "run",
            "construction_events_path": str(events_path),
            "working_blueprint_path": str(blueprint_path),
            "dataset_profile": {},
        }
    )

    report = Path(result["task_screening_report_path"])
    assert report == tmp_path / "task_plans" / "v1.json"
    assert json.loads(report.read_text(encoding="utf-8")) == plan
    snapshot = json.loads(events_path.read_text(encoding="utf-8"))
    assert snapshot["event_type"] == "artifact_snapshot"
    assert snapshot["artifact_path"] == "task_plans/v1.json"
    assert snapshot["artifact_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert snapshot["source"] == "initial_model_output"


def test_task_plan_reconciliation_failure_does_not_write_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    blueprint_path.write_text(json.dumps({"tasks": {"view": _task()}}), encoding="utf-8")
    error = TaskPlanReviewRequired({"tasks": {}}, ["reconciliation: invalid reviewed Task Plan"])

    def fail_generate_task_plan(*_args):
        raise error

    monkeypatch.setattr(capability_agent, "generate_task_plan", fail_generate_task_plan)

    with pytest.raises(TaskPlanReviewRequired):
        capability_agent._task_plan_generate_node(
            {
                "run_id": "run",
                "construction_events_path": str(events_path),
                "working_blueprint_path": str(blueprint_path),
                "dataset_profile": {},
            }
        )

    assert not (tmp_path / "task_plans").exists()


def test_sample_count_revision_records_all_redistributed_tasks(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    blueprint_path = tmp_path / "blueprint.json"
    tasks = {
        "view": _task(category="image_acquisition_and_basic_information"),
        "report": _task("free_text", category="report_generation"),
    }
    blueprint_path.write_text(
        json.dumps({"benchmark_info": {"sample_count": 5}, "tasks": tasks}), encoding="utf-8"
    )
    state = {
        "run_id": "run",
        "construction_events_path": str(events_path),
        "working_blueprint_path": str(blueprint_path),
        "benchmark_info": {"sample_count": 5},
        "draft_sample_counts": {"view": 2, "report": 3},
    }
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "view 改为 4")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "adjust", "counts": {"view": 4}},
    )

    result = capability_agent._sample_count_review_node(state)

    assert result["draft_sample_counts"] == {"view": 4, "report": 1}
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[1]["modified_object_ids"] == ["report", "view"]
    assert construction_report(events_path)["human_stages"]["sample_count"] == {
        "confirmed": 0,
        "revision_rounds": 1,
        "modified_objects": 2,
    }


def test_partial_sample_count_adjustment_redistributes_unspecified_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "adjust", "counts": {"view": 4}},
    )

    tasks = {
        "view": _task(category="image_acquisition_and_basic_information"),
        "report": _task("free_text", category="report_generation"),
    }
    action = capability_agent._sample_count_review_action(
        "view 改为 4",
        {"view": 2, "report": 2},
        5,
        tasks,
    )

    assert action == {"action": "adjust", "counts": {"view": 4}}
    assert capability_agent.distribute_weighted_sample_counts(tasks, 5, action["counts"]) == {
        "view": 4,
        "report": 1,
    }
