import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import medbenchagent.agents.capability as capability_agent
import medbenchagent.agents.generation as generation_agent
import medbenchagent.agents.specification as specification_agent
import medbenchagent.workflow.graph as workflow
from medbenchagent.agents import (
    compile_audit_agent,
    compile_capability_agent,
    compile_clinical_design_agent,
    compile_generation_agent,
    compile_specification_agent,
)
from medbenchagent.agents.contracts import (
    AUDIT_AGENT,
    CAPABILITY_AGENT,
    CLINICAL_DESIGN_AGENT,
    GENERATION_AGENT,
    SPECIFICATION_AGENT,
)
from medbenchagent.contracts import GenerationResult
from medbenchagent.generation import graph as generation_workflow
from medbenchagent.rag import graph as rag_workflow
from medbenchagent.run_artifacts import initialize_run


def _task(
    response_format: str = "letter",
    category: str = "image_acquisition_and_basic_information",
) -> dict:
    task = {
        "description": "Identify view",
        "require": "View annotation",
        "task_type": "close_vqa",
        "response_format": response_format,
        "metric": "accuracy",
        "category": category,
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


def test_compiled_graph_exposes_rag_and_generation_as_expandable_subgraphs() -> None:
    graph = workflow.compile_benchmark_graph().get_graph()
    conditional_edges = {(edge.source, edge.target) for edge in graph.edges if edge.conditional}

    assert conditional_edges == set()
    assert set(graph.nodes) == {
        "__start__",
        "__end__",
        "run_initialize",
        SPECIFICATION_AGENT,
        CAPABILITY_AGENT,
        CLINICAL_DESIGN_AGENT,
        "rag_design_export",
        GENERATION_AGENT,
        "generation_summary",
        AUDIT_AGENT,
        "run_finalize",
    }

    expanded = workflow.compile_benchmark_graph().get_graph(xray=True)
    assert any("benchmark_extract" in node for node in expanded.nodes)
    assert any("dataset_recommend" in node for node in expanded.nodes)
    assert any("rag_prepare" in node for node in expanded.nodes)
    assert any("generation_request" in node for node in expanded.nodes)
    assert any("generation_initialize" in node for node in expanded.nodes)
    assert any("audit" in node for node in expanded.nodes)

    rag_graph = rag_workflow.compile_rag_designs_subgraph()
    assert rag_graph.builder.input_schema is rag_workflow.RagDesignInputState
    assert rag_graph.builder.output_schema is rag_workflow.RagDesignOutputState
    assert set(rag_graph.builder.output_schema.__annotations__) == {"working_blueprint_path"}


def test_role_agents_expose_strict_input_and_output_contracts() -> None:
    agents = (
        compile_specification_agent(),
        compile_capability_agent(),
        compile_clinical_design_agent(),
        compile_generation_agent(),
        compile_audit_agent(),
    )
    for agent in agents:
        assert agent.builder.input_schema is not agent.builder.state_schema
        assert agent.builder.output_schema is not agent.builder.state_schema

    assert set(agents[0].builder.output_schema.__annotations__) == {
        "user_requirement",
        "benchmark_info",
        "working_blueprint_path",
    }
    assert set(agents[1].builder.output_schema.__annotations__) == {
        "working_blueprint_path",
        "selected_dataset_id",
        "dataset_profile",
        "dataset_access_manifest",
        "task_screening_report_path",
        "selected_material_files",
    }
    assert set(agents[2].builder.output_schema.__annotations__) == {"working_blueprint_path"}
    assert set(agents[3].builder.output_schema.__annotations__) == {
        "raw_output_path",
        "provenance_output_path",
        "item_count",
        "task_counts",
        "generation_shortfalls",
    }
    assert set(agents[4].builder.output_schema.__annotations__) == {"audit_artifacts"}


def test_root_graph_exposes_only_declared_input_and_output_fields() -> None:
    graph = workflow.compile_benchmark_graph()

    assert graph.builder.input_schema is workflow.BenchmarkInputState
    assert set(graph.builder.input_schema.__annotations__) == {
        "messages",
        "user_requirement",
        "seed",
        "max_tasks_per_image",
        "question_temperature",
    }
    assert graph.builder.output_schema is workflow.BenchmarkOutputState
    assert set(graph.builder.output_schema.__annotations__) == {
        "benchmark_info",
        "run_id",
        "selected_dataset_id",
        "working_blueprint_path",
        "task_screening_report_path",
        "raw_output_path",
        "item_count",
        "task_counts",
        "generation_shortfalls",
        "provenance_output_path",
        "rag_designs_path",
        "construction_events_path",
        "construction_report_path",
        "manifest_path",
        "benchmark_id",
        "audit_artifacts",
    }
    assert not {
        "benchmark_feedback",
        "benchmark_confirmed",
        "dataset_recommendation",
        "dataset_confirmed",
        "draft_task_plan",
        "pending_task_feedback",
        "task_plan_confirmed",
        "draft_sample_counts",
        "sample_counts_confirmed",
        "material_selection_error",
        "rag_task_queue",
        "rag_task_feedback",
        "rag_current_task_id",
        "rag_completed_count",
        "rag_total_count",
        "designs_confirmed",
        "generation_request",
        "generation_stage",
        "generation_cursor",
        "generation_progress_path",
        "generation_rejections",
        "generation_exhausted",
    } & set(workflow.BenchmarkState.__annotations__)


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


def test_confirmed_benchmark_writes_preallocated_blueprint_path(tmp_path: Path, monkeypatch) -> None:
    run = initialize_run(tmp_path / "runs")
    monkeypatch.setattr(specification_agent, "interrupt", lambda _payload: "confirm")
    monkeypatch.setattr(
        specification_agent,
        "_benchmark_review_action",
        lambda *_args: {"action": "confirm"},
    )

    result = specification_agent._benchmark_confirm_node(
        {
            "benchmark_info": {"disease": "breast cancer", "sample_count": 1},
            "working_blueprint_path": run["working_blueprint_path"],
        }
    )

    path = Path(result["working_blueprint_path"])
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["benchmark_info"] == {
        "disease": "breast cancer",
        "sample_count": 1,
    }


def test_specification_agent_preserves_interrupt_payload_and_resume(
    tmp_path: Path, monkeypatch
) -> None:
    run = initialize_run(tmp_path / "runs")
    benchmark_info = {"disease": "breast cancer", "sample_count": 1}
    monkeypatch.setattr(
        specification_agent,
        "extract_benchmark_info",
        lambda *_args: benchmark_info,
    )
    monkeypatch.setattr(specification_agent, "missing_fields", lambda _info: [])
    monkeypatch.setattr(
        specification_agent,
        "_benchmark_review_action",
        lambda *_args: {"action": "confirm"},
    )
    graph = compile_specification_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "specification-resume"}}

    interrupted = graph.invoke(
        {
            "user_requirement": "Build a benchmark.",
            "run_id": run["run_id"],
            "construction_events_path": run["construction_events_path"],
            "working_blueprint_path": run["working_blueprint_path"],
        },
        config,
    )

    assert interrupted["__interrupt__"][0].value == {
        "kind": "benchmark_info_confirmation",
        "benchmark_info": benchmark_info,
        "question": "Confirm this BenchmarkInfo, or describe the correction.",
    }
    resumed = graph.invoke(Command(resume="confirm"), config)
    assert resumed == {
        "user_requirement": "Build a benchmark.",
        "benchmark_info": benchmark_info,
        "working_blueprint_path": run["working_blueprint_path"],
    }


def test_dataset_recommendation_enriches_selection_without_path(monkeypatch) -> None:
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


def test_dataset_profile_keeps_blueprint_dataset_id_only(tmp_path: Path, monkeypatch) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    blueprint = {"dataset": {"dataset_id": "vindr-mammo"}, "tasks": {}}
    blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")
    monkeypatch.setattr(capability_agent, "load_dataset_catalog", _datasets)
    monkeypatch.setattr(
        capability_agent,
        "resolve_annotation_path",
        lambda _dataset: tmp_path / "annotation.jsonl",
    )
    monkeypatch.setattr(
        capability_agent,
        "prepare_annotation_dataset",
        lambda *_args: (
            {"schema_version": "task-capability-profile.v1", "total_images": 2},
            {
                "annotation_path": str(tmp_path / "annotation.jsonl"),
                "annotation_sha256": "a" * 64,
                "cache_format_version": "annotation-cache.v3",
            },
        ),
    )

    result = capability_agent._dataset_profile_node(
        {"selected_dataset_id": "vindr-mammo", "working_blueprint_path": str(blueprint_path)}
    )

    assert json.loads(blueprint_path.read_text(encoding="utf-8")) == blueprint
    assert result["dataset_profile"]["total_images"] == 2
    assert "annotation_path" not in result["dataset_access_manifest"]


def test_dataset_review_supports_confirm_and_available_reselection(monkeypatch) -> None:
    monkeypatch.setattr(capability_agent, "complete_json", lambda *_args: {"action": "confirm"})
    assert capability_agent._dataset_review_action("确认", _datasets()) == {"action": "confirm"}

    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "select", "dataset_id": "vindr-mammo"},
    )
    assert capability_agent._dataset_review_action("使用 VinDr", _datasets()) == {
        "action": "select",
        "dataset_id": "vindr-mammo",
    }


def test_dataset_review_rejects_unavailable_reselection(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "select", "dataset_id": "future"},
    )

    with pytest.raises(capability_agent.DatasetCatalogError, match="not available"):
        capability_agent._dataset_review_action("使用 Future Dataset", _datasets())


def test_dataset_review_requires_confirmation_after_reselection(monkeypatch) -> None:
    monkeypatch.setattr(capability_agent, "load_dataset_catalog", _datasets)
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "使用 VinDr")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"action": "select", "dataset_id": "vindr-mammo"},
    )
    state = {
        "dataset_recommendation": {
            "dataset_id": "vindr-mammo",
            "name": "VinDr-Mammo",
            "description": "Mammography annotations.",
            "reason": "Matches.",
        }
    }

    result = capability_agent._dataset_review_node(state)

    assert result["dataset_confirmed"] is False
    assert result["selected_dataset_id"] == ""


def test_dataset_profile_resolves_confirmed_id_internally(tmp_path: Path, monkeypatch) -> None:
    expected_path = tmp_path / "annotation.jsonl"
    monkeypatch.setattr(capability_agent, "load_dataset_catalog", _datasets)
    monkeypatch.setattr(capability_agent, "resolve_annotation_path", lambda _dataset: expected_path)

    def prepare(path, _cache_root):
        assert path == expected_path
        return {"total_images": 1, "schema_version": "task-capability-profile.v1"}, {
            "annotation_path": str(expected_path),
            "duckdb_path": "dataset.duckdb",
            "annotation_sha256": "a" * 64,
            "cache_format_version": "annotation-cache.v3",
        }

    monkeypatch.setattr(capability_agent, "prepare_annotation_dataset", prepare)

    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "dataset_id": "vindr-mammo",
                    "dataset_name": "VinDr-Mammo",
                    "description": "Mammography annotations.",
                }
            }
        ),
        encoding="utf-8",
    )
    result = capability_agent._dataset_profile_node(
        {"selected_dataset_id": "vindr-mammo", "working_blueprint_path": str(blueprint_path)}
    )

    assert result["dataset_profile"] == {
        "total_images": 1,
        "schema_version": "task-capability-profile.v1",
    }
    assert result["dataset_access_manifest"] == {
        "duckdb_path": "dataset.duckdb",
        "annotation_sha256": "a" * 64,
        "cache_format_version": "annotation-cache.v3",
    }
    assert "annotation_path" not in result


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

    capability_agent._apply_task_plan_node(state)

    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    assert list(tasks) == ["view"]
    assert tasks["view"]["supporting_fields"] == ["annotation.view"]


def test_rag_processes_one_blueprint_task_and_persists_it(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps({"benchmark_info": {"language": "en"}, "tasks": {"view": _task()}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(
        rag_workflow,
        "design_task",
        lambda _rag, _info, _task_id, _task, dataset_profile, feedback="": _rag_fields(),
    )
    state = {
        "working_blueprint_path": str(path),
        "benchmark_info": {"language": "en"},
        "selected_material_files": ["imaging_reporting_and_classification/guide.pdf"],
        "dataset_profile": {"fields": {}},
        "rag_task_queue": ["view"],
        "rag_task_feedback": {},
        "rag_completed_count": 0,
    }

    result = rag_workflow._rag_design_task_node(state)

    written = json.loads(path.read_text(encoding="utf-8"))["tasks"]["view"]
    assert written["question_templates"] == ["Which view is shown?"]
    assert result["rag_task_queue"] == []
    assert result["rag_completed_count"] == 1


def test_material_selection_uses_exact_scanned_paths_and_reprompts_on_invalid_choice(
    monkeypatch
) -> None:
    materials = [
        {
            "category": "imaging_reporting_and_classification",
            "category_label": "影像报告与分类标准",
            "file": "imaging_reporting_and_classification/guide.pdf",
        }
    ]
    monkeypatch.setattr(capability_agent, "discover_medical_materials", lambda _root: materials)
    monkeypatch.setattr(capability_agent, "interrupt", lambda _payload: "选择指南")
    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"selected_files": [materials[0]["file"]]},
    )

    selected = capability_agent._materials_choose_node({})

    assert selected == {
        "selected_material_files": [materials[0]["file"]],
        "material_selection_error": "",
    }

    monkeypatch.setattr(
        capability_agent,
        "complete_json",
        lambda *_args: {"selected_files": ["missing.pdf"]},
    )
    invalid = capability_agent._materials_choose_node({})

    assert invalid["selected_material_files"] == []
    assert "Unknown medical PDF selection" in invalid["material_selection_error"]


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

    result = rag_workflow._rag_prepare_node({"working_blueprint_path": str(path)})

    assert result["rag_task_queue"] == ["view"]
    assert result["rag_total_count"] == 1


def test_design_review_action_supports_multiple_targeted_revisions(monkeypatch) -> None:
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
        "run_id": "00000000-0000-0000-0000-000000000000",
        "selected_dataset_id": "vindr-mammo",
        "construction_events_path": str(tmp_path / "construction_events.jsonl"),
        "dataset_access_manifest": {"duckdb_path": "dataset.duckdb"},
        "seed": 3,
        "max_tasks_per_image": 2,
        "question_temperature": 0.4,
    }

    result = generation_agent._generation_request_node(state)
    request = result["generation_request"]

    assert request["tasks"] == {"view": task}
    assert "task_specs" not in request
    assert "question_designs" not in request
    assert "approved_task_plans" not in request


def test_generation_request_is_private_to_generation_agent_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "blueprint.json"
    task = _task() | _rag_fields() | {"sample_count": 1}
    path.write_text(
        json.dumps({"benchmark_info": {"sample_count": 1}, "tasks": {"view": task}}),
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
            "generation_exhausted": False,
        },
    )
    monkeypatch.setattr(
        generation_workflow,
        "finalize_generation",
        lambda _request, _state: GenerationResult(
            raw_output_path=str(tmp_path / "raw.jsonl"),
            provenance_output_path=str(tmp_path / "provenance.jsonl"),
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
        "provenance_output_path",
        "item_count",
        "task_counts",
        "generation_shortfalls",
    }
    assert "generation_request" not in result
    assert any(
        "generation_request" in snapshot.values
        for snapshot in graph.get_state_history(config)
    )


def test_generation_subgraph_checkpoints_each_completed_record(tmp_path: Path, monkeypatch) -> None:
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
            update |= {"generation_stage_index": 3, "generation_stage": "", "generation_exhausted": True}
        return update

    def finalize(_request: dict, state: dict) -> GenerationResult:
        return GenerationResult(
            raw_output_path=str(tmp_path / "benchmark_items.raw.jsonl"),
            item_count=state["generation_completed_counts"]["view"],
            task_counts=state["generation_completed_counts"],
        )

    monkeypatch.setattr(generation_workflow, "initialize_generation", initialize)
    monkeypatch.setattr(generation_workflow, "advance_generation", advance)
    monkeypatch.setattr(generation_workflow, "finalize_generation", finalize)
    checkpointer = MemorySaver()
    graph = generation_workflow.compile_generation_subgraph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "generation-test"}, "recursion_limit": 20}

    result = graph.invoke({"generation_request": {}}, config)
    history = list(graph.get_state_history(config))

    assert calls == 2
    assert result == {
        "raw_output_path": str(tmp_path / "benchmark_items.raw.jsonl"),
        "provenance_output_path": "",
        "item_count": 2,
        "task_counts": {"view": 2},
        "generation_shortfalls": {"view": 0},
    }
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


def test_rag_subgraph_checkpoints_private_queue_state(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"language": "en"},
                "tasks": {"view": _task() | {"sample_count": 1}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(
        rag_workflow,
        "design_task",
        lambda _rag, _info, _task_id, _task, dataset_profile, feedback="": _rag_fields(),
    )
    monkeypatch.setattr(rag_workflow, "interrupt", lambda _payload: "确认")
    monkeypatch.setattr(rag_workflow, "complete_json", lambda *_args: {"action": "confirm"})
    checkpointer = MemorySaver()
    graph = rag_workflow.compile_rag_designs_subgraph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "rag-test"}, "recursion_limit": 20}

    result = graph.invoke(
        {
            "benchmark_info": {"language": "en"},
            "working_blueprint_path": str(path),
            "selected_material_files": ["imaging_reporting_and_classification/guide.pdf"],
            "dataset_profile": {"fields": {}},
        },
        config,
    )
    history = list(graph.get_state_history(config))

    assert result == {"working_blueprint_path": str(path)}
    private_states = [
        snapshot.values
        for snapshot in history
        if snapshot.values.get("rag_current_task_id") == "view"
    ]
    assert private_states
    assert private_states[0]["rag_completed_count"] == 1
    assert private_states[0]["rag_task_queue"] == []


def test_rag_subgraph_streams_custom_progress(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "blueprint.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_info": {"language": "en"},
                "tasks": {"view": _task() | {"sample_count": 1}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_workflow, "LocalPdfRag", lambda *_args: object())
    monkeypatch.setattr(
        rag_workflow,
        "design_task",
        lambda _rag, _info, _task_id, _task, dataset_profile, feedback="": _rag_fields(),
    )
    monkeypatch.setattr(rag_workflow, "interrupt", lambda _payload: "确认")
    monkeypatch.setattr(rag_workflow, "complete_json", lambda *_args: {"action": "confirm"})

    events = list(
        rag_workflow.compile_rag_designs_subgraph().stream(
            {
                "benchmark_info": {"language": "en"},
                "working_blueprint_path": str(path),
                "selected_material_files": ["imaging_reporting_and_classification/guide.pdf"],
                "dataset_profile": {"fields": {}},
            },
            stream_mode="custom",
        )
    )

    assert events == [
        {
            "kind": "rag_progress",
            "task_id": "view",
            "completed": 1,
            "total": 1,
            "remaining": 0,
        }
    ]


def test_generation_subgraph_streams_custom_progress(tmp_path: Path, monkeypatch) -> None:
    def initialize(_request: dict) -> dict:
        return {
            "generation_target_counts": {"view": 1},
            "generation_completed_counts": {"view": 0},
            "generation_stage_index": 0,
            "generation_stage": "vqa",
            "generation_cursor": 0,
            "generation_progress_path": str(tmp_path / "benchmark_items.progress.jsonl"),
            "generation_rejections": {"view": 0},
            "generation_exhausted": False,
        }

    def advance(_request: dict, _state: dict) -> dict:
        return {
            "generation_completed_counts": {"view": 1},
            "generation_stage_index": 3,
            "generation_stage": "",
            "generation_cursor": 1,
            "generation_exhausted": True,
        }

    def finalize(_request: dict, _state: dict) -> GenerationResult:
        return GenerationResult(
            raw_output_path=str(tmp_path / "benchmark_items.raw.jsonl"),
            item_count=1,
            task_counts={"view": 1},
        )

    monkeypatch.setattr(generation_workflow, "initialize_generation", initialize)
    monkeypatch.setattr(generation_workflow, "advance_generation", advance)
    monkeypatch.setattr(generation_workflow, "finalize_generation", finalize)

    events = list(
        generation_workflow.compile_generation_subgraph().stream(
            {"generation_request": {}}, stream_mode="custom"
        )
    )

    assert events == [
        {
            "kind": "generation_progress",
            "stage": "",
            "cursor": 1,
            "completed_counts": {"view": 1},
            "target_counts": {"view": 1},
            "rejections": {"view": 0},
            "exhausted": True,
        }
    ]
