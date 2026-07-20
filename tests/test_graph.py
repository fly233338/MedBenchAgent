import json
from pathlib import Path

import medbenchagent.workflow.graph as workflow


def _task(response_format: str = "letter") -> dict:
    task = {
        "description": "Identify view",
        "require": "View annotation",
        "task_type": "close_vqa",
        "response_format": response_format,
        "metric": "accuracy",
        "supporting_fields": ["annotation.view"],
        "dataset_support_reason": "Available.",
    }
    return task


def _rag_fields(response_format: str = "letter") -> dict:
    fields = {
        "question_templates": ["Which view is shown?"],
        "generation_logic": "Use the approved annotation.",
        "medical_task_description": "This assesses recognition; references support terminology.",
        "citations": [],
    }
    if response_format == "letter":
        fields["options"] = ["CC", "MLO"]
    return fields


def test_compiled_graph_exposes_rag_task_loop() -> None:
    graph = workflow.compile_benchmark_graph().get_graph()
    conditional_edges = {(edge.source, edge.target) for edge in graph.edges if edge.conditional}

    assert conditional_edges == {
        ("benchmark_confirm", "dataset_profile"),
        ("benchmark_confirm", "benchmark_extract"),
        ("dataset_profile", "task_plan_generate"),
        ("dataset_profile", "dataset_profile"),
        ("task_plan_review", "apply_task_plan"),
        ("task_plan_review", "task_plan_revise"),
        ("rag_design_task", "rag_design_task"),
        ("rag_design_task", "design_review"),
        ("design_review", "generation_request"),
        ("design_review", "rag_design_task"),
    }


def test_apply_task_plan_updates_blueprint_as_single_task_source(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps({"tasks": {"view": _task(), "diagnosis": _task()}}), encoding="utf-8"
    )
    state = {
        "working_blueprint_path": str(path),
        "dataset_profile": {"fields": {"annotation.view": {}}},
        "draft_task_plan": {
            "tasks": {
                "view": {
                    "decision": "retain",
                    "supporting_fields": ["annotation.view"],
                    "reason": "Available.",
                },
                "diagnosis": {"decision": "drop", "supporting_fields": [], "reason": "Missing."},
            }
        },
    }

    workflow._apply_task_plan_node(state)

    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    assert list(tasks) == ["view"]
    assert tasks["view"]["supporting_fields"] == ["annotation.view"]


def test_rag_processes_one_blueprint_task_and_persists_it(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps({"benchmark_info": {"language": "en"}, "tasks": {"view": _task()}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(
        workflow,
        "design_task",
        lambda _rag, _info, _task_id, _task, feedback="": _rag_fields(),
    )
    state = {
        "working_blueprint_path": str(path),
        "benchmark_info": {"language": "en"},
        "materials_root": str(tmp_path),
        "rag_task_queue": ["view"],
        "rag_task_feedback": {},
        "rag_completed_count": 0,
    }

    result = workflow._rag_design_task_node(state)

    written = json.loads(path.read_text(encoding="utf-8"))["tasks"]["view"]
    assert written["question_templates"] == ["Which view is shown?"]
    assert result["rag_task_queue"] == []
    assert result["rag_completed_count"] == 1


def test_design_review_action_supports_multiple_targeted_revisions(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "complete_json",
        lambda *_args: {
            "action": "revise",
            "revisions": [
                {"task_id": "view", "feedback": "Shorter wording."},
                {"task_id": "report", "feedback": "Clarify capability."},
            ],
        },
    )

    result = workflow._design_review_action("修改两个任务", {"view": {}, "report": {}})

    assert [revision["task_id"] for revision in result["revisions"]] == ["view", "report"]


def test_generation_request_uses_blueprint_task_snapshot_without_design_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blueprint.json"
    task = _task() | _rag_fields()
    path.write_text(
        json.dumps({"benchmark_info": {"sample_count": 2}, "tasks": {"view": task}}),
        encoding="utf-8",
    )
    state = {
        "working_blueprint_path": str(path),
        "benchmark_info": {"sample_count": 2},
        "dataset_access_manifest": {"duckdb_path": "dataset.duckdb"},
        "seed": 3,
        "balance_answer_positions": True,
    }

    result = workflow._generation_request_node(state)
    request = result["generation_request"]

    assert request["tasks"] == {"view": task}
    assert "task_specs" not in request
    assert "question_designs" not in request
    assert "approved_task_plans" not in request
