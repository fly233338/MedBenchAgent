import hashlib
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import medbenchagent.agents.capability as capability_agent
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
from medbenchagent.construction import construction_report
from medbenchagent.contracts import GenerationResult
from medbenchagent.generation import graph as generation_workflow
from medbenchagent.rag import graph as rag_workflow
from medbenchagent.run_artifacts import initialize_run, write_versioned_json


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
        "supporting_fields": ["image.annotation.view"],
        "dataset_support_reason": "Available.",
    }
    return task


def _rag_fields(response_format: str = "letter") -> dict:
    fields = {
        "question_templates": ["Which view is shown?"],
        "generation_logic": "Use the approved supporting fields.",
        "medical_task_description": "This assesses recognition; references support terminology.",
        "citations": [],
    }
    if response_format == "letter":
        fields["options"] = ["CC", "MLO"]
        fields["answer_field"] = "image.annotation.view"
        fields["answer_value_mapping"] = [
            {"raw_value": "CC", "option": "CC"},
            {"raw_value": "MLO", "option": "MLO"},
        ]
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


def test_rag_design_export_uses_latest_versioned_artifact(tmp_path: Path) -> None:
    first = write_versioned_json(tmp_path, "rag_designs", {"tasks": {"view": {"version": 1}}})
    second = write_versioned_json(tmp_path, "rag_designs", {"tasks": {"view": {"version": 2}}})

    result = workflow._rag_design_export_node({"run_directory": str(tmp_path)})

    assert first["path"] != second["path"]
    assert result["rag_designs_path"] == second["path"]


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
    blueprint_path = Path(resumed["working_blueprint_path"])
    assert blueprint_path.is_file()
    assert json.loads(blueprint_path.read_text(encoding="utf-8"))["benchmark_info"] == benchmark_info
    human_events = [
        event
        for event in map(
            json.loads,
            Path(run["construction_events_path"]).read_text(encoding="utf-8").splitlines(),
        )
        if event["event_type"] == "human_interaction"
    ]
    assert [(event["action"], event["user_input"]) for event in human_events] == [
        ("submit", "Build a benchmark."),
        ("confirm", "confirm"),
    ]


def test_specification_agent_records_two_applied_revisions_before_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    run = initialize_run(tmp_path / "runs")
    infos = iter(
        [
            {"language": "en", "sample_count": 10},
            {"language": "zh", "sample_count": 10},
            {"language": "zh", "sample_count": 20},
        ]
    )
    actions = iter(
        [
            {"action": "revise", "feedback": "use Chinese"},
            {"action": "revise", "feedback": "make 20 questions"},
            {"action": "confirm"},
        ]
    )
    monkeypatch.setattr(specification_agent, "extract_benchmark_info", lambda *_args: next(infos))
    monkeypatch.setattr(specification_agent, "missing_fields", lambda _info: [])
    monkeypatch.setattr(specification_agent, "_benchmark_review_action", lambda *_args: next(actions))
    graph = compile_specification_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "specification-revisions"}}
    graph.invoke(
        {
            "user_requirement": "Build a benchmark.",
            "run_id": run["run_id"],
            "construction_events_path": run["construction_events_path"],
            "working_blueprint_path": run["working_blueprint_path"],
        },
        config,
    )
    graph.invoke(Command(resume="把语言改成中文"), config)
    graph.invoke(Command(resume="题量改成 20"), config)
    graph.invoke(Command(resume="确认"), config)

    events = [
        json.loads(line)
        for line in Path(run["construction_events_path"]).read_text(encoding="utf-8").splitlines()
    ]
    review_events = [
        event
        for event in events
        if event["event_type"] == "human_interaction" and event.get("action") != "submit"
    ]
    applied = [event for event in events if event["event_type"] == "revision_applied"]
    assert [event["user_input"] for event in review_events] == [
        "把语言改成中文",
        "题量改成 20",
        "确认",
    ]
    assert [event["modified_object_ids"] for event in applied] == [
        ["language"],
        ["sample_count"],
    ]
    assert construction_report(run["construction_events_path"])["human_stages"][
        "benchmark_info"
    ] == {"confirmed": 1, "revision_rounds": 2, "modified_objects": 2}


def test_dataset_recommendation_and_profile_keep_paths_private(
    tmp_path: Path, monkeypatch
) -> None:
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
        return {"schema_version": "task-capability-profile.v2", "total_images": 2}, {
            "annotation_path": str(expected_path),
            "duckdb_path": "dataset.duckdb",
            "annotation_sha256": "a" * 64,
            "cache_format_version": "annotation-cache.v5",
        }

    monkeypatch.setattr(capability_agent, "prepare_annotation_dataset", prepare)

    profile_result = capability_agent._dataset_profile_node(
        {"selected_dataset_id": "vindr-mammo", "working_blueprint_path": str(blueprint_path)}
    )

    assert json.loads(blueprint_path.read_text(encoding="utf-8")) == blueprint
    assert profile_result["dataset_profile"] == {
        "schema_version": "task-capability-profile.v2",
        "total_images": 2,
    }
    assert profile_result["dataset_access_manifest"] == {
        "duckdb_path": "dataset.duckdb",
        "annotation_sha256": "a" * 64,
        "cache_format_version": "annotation-cache.v5",
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
            "category_label": "Standard",
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


def test_task_plan_revision_records_only_tasks_with_actual_changes(tmp_path: Path, monkeypatch) -> None:
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
        lambda *_args, **kwargs: _rag_fields()
        | {"question_templates": ["Name the view."] if kwargs["feedback"] else ["Which view?"]},
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
    assert applied[0]["artifact_sha256"] == hashlib.sha256(
        (tmp_path / "rag_designs" / "v2.json").read_bytes()
    ).hexdigest()
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


def test_generation_agent_keeps_request_snapshot_private(
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
    requests = [
        snapshot.values["generation_request"]
        for snapshot in graph.get_state_history(config)
        if "generation_request" in snapshot.values
    ]
    assert requests
    assert requests[0]["tasks"] == {"view": task}
    assert not {"task_specs", "question_designs", "approved_task_plans"} & requests[0].keys()


def test_generation_subgraph_checkpoints_and_streams_progress(
    tmp_path: Path, monkeypatch
) -> None:
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
        lambda _rag, _info, _task_id, _task, dataset_profile, dataset_access_manifest, feedback="": _rag_fields(),
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
