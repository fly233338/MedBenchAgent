import hashlib
import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from medbenchagent.construction import construction_report
from medbenchagent.rag import graph as rag_workflow


from graph_test_support import _rag_fields, _task


def test_medical_align_review_supports_multiple_targeted_revisions(monkeypatch) -> None:
    monkeypatch.setattr(
        rag_workflow,
        "complete_json",
        lambda *_args: {
            "action": "revise",
            "revisions": [
                {"task_id": "view", "feedback": "Shorter wording."},
                {"task_id": "report", "feedback": "Clarify capability."},
            ],
        },
    )

    result = rag_workflow._design_review_action("修改两个任务", {"view": {}, "report": {}})

    assert [revision["task_id"] for revision in result["revisions"]] == ["view", "report"]


def test_rag_design_revision_records_actual_changed_tasks(tmp_path: Path, monkeypatch) -> None:
    events_path = tmp_path / "construction_events.jsonl"
    events_path.touch()
    blueprint_path = tmp_path / "blueprint.json"
    task = _task() | {"sample_count": 1}
    blueprint_path.write_text(
        json.dumps({"benchmark_info": {"language": "en"}, "tasks": {"view": task}}),
        encoding="utf-8",
    )
    state = {
        "run_id": "run",
        "selected_dataset_id": "vindr-mammo",
        "construction_events_path": str(events_path),
        "working_blueprint_path": str(blueprint_path),
        "benchmark_info": {"language": "en"},
        "selected_material_files": ["Standard/guide.pdf"],
        "dataset_profile": {"fields": {}},
        "dataset_access_manifest": {"duckdb_path": "unused"},
    }
    monkeypatch.setattr(rag_workflow, "interrupt", lambda _payload: "缩短 view 题干")
    monkeypatch.setattr(
        rag_workflow,
        "complete_json",
        lambda *_args: {
            "action": "revise",
            "revisions": [{"task_id": "view", "feedback": "Shorter wording."}],
        },
    )
    monkeypatch.setattr(rag_workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(
        rag_workflow,
        "design_task",
        lambda *_args, **kwargs: (
            _rag_fields()
            | {"question_templates": ["Name the view."] if kwargs["feedback"] else ["Which view?"]}
        ),
    )

    rag_workflow._rag_design_task_node(state | {"rag_task_queue": ["view"]})

    review = rag_workflow._design_review_node(state)
    rag_workflow._rag_design_task_node(state | review)

    initial = json.loads((tmp_path / "rag_designs" / "v1.json").read_text(encoding="utf-8"))
    revised = json.loads((tmp_path / "rag_designs" / "v2.json").read_text(encoding="utf-8"))
    assert initial["tasks"]["view"]["question_templates"] == ["Which view?"]
    assert revised["tasks"]["view"]["question_templates"] == ["Name the view."]

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    applied = [event for event in events if event["event_type"] == "revision_applied"]
    assert applied[0]["modified_object_ids"] == ["view"]
    assert applied[0]["artifact_path"] == "rag_designs/v2.json"
    assert (
        applied[0]["artifact_sha256"]
        == hashlib.sha256((tmp_path / "rag_designs" / "v2.json").read_bytes()).hexdigest()
    )
    assert construction_report(events_path)["human_stages"]["rag_design"] == {
        "confirmed": 0,
        "revision_rounds": 1,
        "modified_objects": 1,
    }


def test_rag_design_draft_is_written_before_human_review(tmp_path: Path, monkeypatch) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    task = _task() | {"sample_count": 1}
    blueprint_path.write_text(
        json.dumps({"benchmark_info": {"language": "en"}, "tasks": {"view": task}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(rag_workflow, "design_task", lambda *_args, **_kwargs: _rag_fields())

    rag_workflow._rag_design_task_node(
        {
            "run_id": "run",
            "selected_dataset_id": "vindr-mammo",
            "working_blueprint_path": str(blueprint_path),
            "benchmark_info": {"language": "en"},
            "selected_material_files": ["Standard/guide.pdf"],
            "dataset_profile": {"fields": {}},
            "dataset_access_manifest": {"duckdb_path": "unused"},
            "rag_task_queue": ["view"],
        }
    )

    designs = json.loads((tmp_path / "rag_designs" / "v1.json").read_text(encoding="utf-8"))
    assert designs["tasks"]["view"] == _rag_fields()


def test_rag_subgraph_persists_design_and_checkpoints_streamed_progress(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"language": "en"},
                "tasks": {
                    "view": _task() | {"sample_count": 1},
                    "report": _task("free_text") | {"sample_count": 0},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(
        rag_workflow,
        "design_task",
        lambda _rag, _info, _task_id, _task, dataset_profile, dataset_access_manifest, feedback="": (
            _rag_fields()
        ),
    )
    monkeypatch.setattr(rag_workflow, "interrupt", lambda _payload: "确认")
    monkeypatch.setattr(rag_workflow, "complete_json", lambda *_args: {"action": "confirm"})
    checkpointer = MemorySaver()
    graph = rag_workflow.compile_rag_designs_subgraph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "rag-test"}, "recursion_limit": 20}

    events = list(
        graph.stream(
            {
                "benchmark_info": {"language": "en"},
                "working_blueprint_path": str(path),
                "selected_material_files": ["Standard/guide.pdf"],
                "dataset_profile": {"fields": {}},
                "dataset_access_manifest": {"duckdb_path": "unused"},
            },
            config,
            stream_mode="custom",
        )
    )
    history = list(graph.get_state_history(config))

    assert events == [
        {
            "kind": "rag_progress",
            "task_id": "view",
            "completed": 1,
            "total": 1,
            "remaining": 0,
        }
    ]
    written = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    assert written["view"]["question_templates"] == ["Which view is shown?"]
    assert "question_templates" not in written["report"]
    private_states = [
        snapshot.values
        for snapshot in history
        if snapshot.values.get("rag_current_task_id") == "view"
    ]
    assert private_states
    assert private_states[0]["rag_completed_count"] == 1
    assert private_states[0]["rag_task_queue"] == []
