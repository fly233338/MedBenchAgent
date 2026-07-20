import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

import medbenchagent.workflow.graph as workflow
from medbenchagent.contracts import GenerationResult


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
        ("sample_count_review", "materials_root"),
        ("sample_count_review", "sample_count_review"),
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


def test_sample_counts_are_evenly_prepared_and_written_only_after_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"sample_count": 5},
                "tasks": {"view": _task(), "diagnosis": _task()},
            }
        ),
        encoding="utf-8",
    )
    state = {
        "working_blueprint_path": str(path),
        "benchmark_info": {"sample_count": 5},
    }

    prepared = workflow._sample_count_prepare_node(state)
    assert prepared["draft_sample_counts"] == {"view": 3, "diagnosis": 2}
    assert "sample_count" not in json.loads(path.read_text(encoding="utf-8"))["tasks"]["view"]

    monkeypatch.setattr(workflow, "interrupt", lambda _payload: "确认")
    monkeypatch.setattr(workflow, "complete_json", lambda *_args: {"action": "confirm"})
    reviewed = workflow._sample_count_review_node(state | prepared)

    assert reviewed["sample_counts_confirmed"] is True
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    assert {task_id: task["sample_count"] for task_id, task in tasks.items()} == {
        "view": 3,
        "diagnosis": 2,
    }


def test_partial_sample_count_adjustment_redistributes_unspecified_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow,
        "complete_json",
        lambda *_args: {"action": "adjust", "counts": {"view": 4}},
    )

    action = workflow._sample_count_review_action("view 改为 4", {"view": 2, "report": 2}, 5)

    assert action == {"action": "adjust", "counts": {"view": 4}}
    assert workflow.distribute_sample_counts(
        ["view", "report"], 5, action["counts"]
    ) == {"view": 4, "report": 1}


def test_zero_count_tasks_are_skipped_by_rag_queue(tmp_path: Path) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "tasks": {
                    "view": _task() | {"sample_count": 2},
                    "report": _task("free_text") | {"sample_count": 0},
                }
            }
        ),
        encoding="utf-8",
    )

    result = workflow._rag_prepare_node({"working_blueprint_path": str(path)})

    assert result["rag_task_queue"] == ["view"]
    assert result["rag_total_count"] == 1


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
    task = _task() | _rag_fields() | {"sample_count": 2}
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


def test_generation_subgraph_checkpoints_each_completed_record(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def initialize(_request: dict) -> dict:
        return {
            "generation_target_counts": {"view": 2},
            "generation_completed_counts": {"view": 0},
            "generation_cursor": 0,
            "generation_progress_path": str(tmp_path / "benchmark_items.progress.jsonl"),
            "generation_rejections": {},
            "generation_exhausted": False,
        }

    def advance(_request: dict, state: dict) -> dict:
        nonlocal calls
        calls += 1
        return {
            "generation_completed_counts": {"view": calls},
            "generation_cursor": state["generation_cursor"] + 1,
        }

    def finalize(_request: dict, state: dict) -> GenerationResult:
        return GenerationResult(
            raw_output_path=str(tmp_path / "benchmark_items.raw.jsonl"),
            item_count=state["generation_completed_counts"]["view"],
            task_counts=state["generation_completed_counts"],
        )

    monkeypatch.setattr(workflow, "initialize_generation", initialize)
    monkeypatch.setattr(workflow, "advance_generation", advance)
    monkeypatch.setattr(workflow, "finalize_generation", finalize)
    checkpointer = MemorySaver()
    graph = workflow.compile_generation_subgraph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "generation-test"}, "recursion_limit": 20}

    result = graph.invoke({"generation_request": {}}, config)
    history = list(graph.get_state_history(config))

    assert calls == 2
    assert result["generated_item_count"] == 2
    assert result["generation_shortfalls"] == {"view": 0}
    assert len(history) >= 5
