from pathlib import Path


import medbenchagent.workflow.graph as workflow
from medbenchagent.agents import (
    compile_audit_agent,
    compile_task_planning_agent,
    compile_medical_align_agent,
    compile_generation_agent,
    compile_specification_agent,
)
from medbenchagent.agents.contracts import (
    AUDIT_AGENT,
    TASK_PLANNING_AGENT,
    MEDICAL_ALIGN_AGENT,
    GENERATION_AGENT,
    SPECIFICATION_AGENT,
)
from medbenchagent.rag import graph as rag_workflow
from medbenchagent.run_artifacts import write_versioned_json


def test_compiled_graph_exposes_rag_and_generation_as_expandable_subgraphs() -> None:
    graph = workflow.compile_benchmark_graph().get_graph()
    conditional_edges = {(edge.source, edge.target) for edge in graph.edges if edge.conditional}

    assert conditional_edges == set()
    assert set(graph.nodes) == {
        "__start__",
        "__end__",
        "run_initialize",
        SPECIFICATION_AGENT,
        TASK_PLANNING_AGENT,
        MEDICAL_ALIGN_AGENT,
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
        compile_task_planning_agent(),
        compile_medical_align_agent(),
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
