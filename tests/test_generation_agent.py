import json
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from medbenchagent.agents.generation import (
    compile_generation_agent,
)
from medbenchagent.contracts import GenerationResult
from medbenchagent.generation import graph as generation_workflow


from test_generation import _task


def test_generation_agent_keeps_request_snapshot_private(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "blueprint.json"
    task = _task("letter", ["image.annotation.view"], sample_count=1)
    templates = task.pop("question_templates")
    (tmp_path / "templates.json").write_text(json.dumps({"view": templates}), encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"sample_count": 1},
                "allocation": {"allocated_count": 1},
                "tasks": {"view": task},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        generation_workflow,
        "initialize_generation",
        lambda _request: {
            "generation_target_counts": {"view": 1},
            "generation_completed_counts": {"view": 1},
            "generation_stage_index": 3,
            "generation_stage": "",
            "generation_cursor": 1,
            "generation_progress_path": str(tmp_path / "progress.jsonl"),
            "generation_rejections": {},
            "generation_exhausted": True,
        },
    )
    monkeypatch.setattr(
        generation_workflow,
        "finalize_generation",
        lambda _request: GenerationResult(
            raw_output_path=str(tmp_path / "raw.jsonl"),
            item_count=1,
            task_counts={"view": 1},
        ),
    )
    graph = compile_generation_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "generation-agent-private"}}
    result = graph.invoke(
        {
            "working_blueprint_path": str(path),
            "benchmark_info": {"sample_count": 1},
            "run_id": "run",
            "selected_dataset_id": "vindr-mammo",
            "construction_events_path": str(tmp_path / "events.jsonl"),
            "dataset_access_manifest": {"duckdb_path": "dataset.duckdb"},
        },
        config,
    )

    assert set(result) == {
        "raw_output_path",
        "item_count",
        "task_counts",
        "generation_shortfalls",
    }
    assert "generation_request" not in result
    requests = [
        snapshot.values["generation_request"]
        for snapshot in graph.get_state_history(config)
        if "generation_request" in snapshot.values
    ]
    assert requests
    assert requests[0]["tasks"] == {"view": {**task, "question_templates": templates}}
    assert "question_templates" not in json.loads(path.read_text(encoding="utf-8"))["tasks"]["view"]
    assert not {"task_specs", "question_designs", "approved_task_plans"} & requests[0].keys()


def test_generation_subgraph_checkpoints_and_streams_progress(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def initialize(_request: dict) -> dict:
        return {
            "generation_target_counts": {"view": 2},
            "generation_completed_counts": {"view": 0},
            "generation_stage_index": 0,
            "generation_stage": "vqa",
            "generation_cursor": 0,
            "generation_progress_path": str(tmp_path / "benchmark_items.progress.jsonl"),
            "generation_rejections": {},
            "generation_exhausted": False,
        }

    def advance(_request: dict, state: dict) -> dict:
        nonlocal calls
        calls += 1
        update = {
            "generation_completed_counts": {"view": calls},
            "generation_cursor": state["generation_cursor"] + 1,
        }
        if calls == 2:
            update |= {
                "generation_stage_index": 3,
                "generation_stage": "",
                "generation_exhausted": True,
            }
        return update

    def finalize(_request: dict) -> GenerationResult:
        return GenerationResult(
            raw_output_path=str(tmp_path / "benchmark_items.raw.jsonl"),
            item_count=2,
            task_counts={"view": 2},
        )

    monkeypatch.setattr(generation_workflow, "initialize_generation", initialize)
    monkeypatch.setattr(generation_workflow, "advance_generation", advance)
    monkeypatch.setattr(generation_workflow, "finalize_generation", finalize)
    checkpointer = MemorySaver()
    graph = generation_workflow.compile_generation_subgraph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "generation-test"}, "recursion_limit": 20}

    events = list(
        graph.stream(
            {"generation_request": {}},
            config,
            stream_mode="custom",
        )
    )
    history = list(graph.get_state_history(config))

    assert calls == 2
    assert events == [
        {
            "kind": "generation_progress",
            "stage": "vqa",
            "cursor": 1,
            "completed_counts": {"view": 1},
            "target_counts": {"view": 2},
            "rejections": {},
            "exhausted": False,
        },
        {
            "kind": "generation_progress",
            "stage": "",
            "cursor": 2,
            "completed_counts": {"view": 2},
            "target_counts": {"view": 2},
            "rejections": {},
            "exhausted": True,
        },
    ]
    private_states = [
        snapshot.values
        for snapshot in history
        if snapshot.values.get("generation_cursor") == 2
        and snapshot.values.get("generation_completed_counts") == {"view": 2}
    ]
    assert private_states
    assert private_states[0]["generation_progress_path"] == str(
        tmp_path / "benchmark_items.progress.jsonl"
    )
    assert private_states[0]["generation_target_counts"] == {"view": 2}
    assert private_states[0]["generation_stage"] == ""
    assert private_states[0]["generation_rejections"] == {}
    assert private_states[0]["generation_exhausted"] is True
    assert len(history) >= 5
