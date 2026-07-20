from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import medbenchagent.workflow.graph as workflow


def test_graph_reaches_generation_request_with_confirmations(tmp_path: Path, monkeypatch) -> None:
    annotation_path = tmp_path / "annotation.jsonl"
    annotation_path.write_text(
        '{"image_id":"img-1","image_path":"/external/images/img-1.dicom","patient_id":null,"case_id":null,"study_id":"study-1","annotation":{"breast_birads":"BI-RADS 4"}}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        workflow,
        "extract_benchmark_info",
        lambda requirement, existing: {
            "disease": "breast cancer",
            "modality": "mammography",
            "description": "diagnosis benchmark",
            "language": "en",
            "sample_count": 2,
        },
    )
    task_plan_calls = 0

    def fake_generate_task_plan(_blueprint, _profile):
        nonlocal task_plan_calls
        task_plan_calls += 1
        return {
            "tasks": {
                "assessment_category": {
                    "decision": "retain",
                    "supporting_fields": ["annotation.breast_birads"],
                    "reason": "BI-RADS is available.",
                }
            }
        }

    monkeypatch.setattr(workflow, "generate_task_plan", fake_generate_task_plan)
    monkeypatch.setattr(workflow, "apply_task_plan", lambda _path, _plans, _profile: str(tmp_path / "report.json"))

    class FakeRag:
        def __init__(self, *_args):
            pass

    monkeypatch.setattr(workflow, "LocalPdfRag", FakeRag)
    monkeypatch.setattr(
        workflow,
        "design_questions",
        lambda _rag, _info, plans, _feedback: [
            {
                "task_id": plan["task_id"],
                "question_templates": ["What is the BI-RADS category?"],
                "placeholder_bindings": {},
                "answer_list": ["BI-RADS 2", "BI-RADS 4"],
                "answer_field": "annotation.breast_birads",
                "answer_value_mapping": {},
                "validation_rules": [],
                "citations": [],
            }
            for plan in plans
        ],
    )
    raw_path = tmp_path / "benchmark_items.raw.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "generate_raw_benchmark",
        lambda _request: SimpleNamespace(raw_output_path=str(raw_path), item_count=2, task_counts={"assessment_category": 2}),
    )
    monkeypatch.setattr(workflow, "balance_jsonl", lambda _input, output, _seed: output)
    graph = workflow.compile_benchmark_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}
    result = graph.invoke({"messages": [HumanMessage(content="make a benchmark")]}, config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume="yes"), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume=str(annotation_path)), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume="yes"), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume=str(tmp_path / "materials")), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume="yes"), config)
    assert result["ready_for_generation"] is True
    assert result["generation_request"]["target_sample_count"] == 2
    assert result["generation_request"]["approved_task_plans"][0]["supporting_fields"] == [
        "annotation.breast_birads"
    ]
    assert result["generation_request"]["task_specs"][0]["response_format"] == "letter"
    assert result["raw_output_path"] == str(raw_path)
    assert result["benchmark_output_path"].endswith("benchmark_items.jsonl")
    assert task_plan_calls == 1


def test_finalize_uses_raw_file_when_balancing_is_disabled(tmp_path: Path, monkeypatch) -> None:
    raw_path = tmp_path / "benchmark_items.raw.jsonl"
    raw_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "balance_jsonl", lambda *_args: (_ for _ in ()).throw(AssertionError("not called")))

    result = workflow._finalize_benchmark_node(
        {"raw_output_path": str(raw_path), "generation_request": {"balance_answer_positions": False}}
    )

    assert result == {"benchmark_output_path": str(raw_path)}


def test_generation_request_keeps_non_closed_tasks(tmp_path: Path) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(
        '{"tasks":{'
        '"detection":{"task_type":"open_vqa","response_format":"bbox","description":"Locate"},'
        '"report_generation":{"task_type":"open_vqa","response_format":"free_text","description":"Report"}'
        '}}',
        encoding="utf-8",
    )
    state = {
        "working_blueprint_path": str(blueprint_path),
        "benchmark_info": {"sample_count": 2},
        "dataset_access_manifest": {},
        "task_plans": [
            {"task_id": "detection", "decision": "retain", "supporting_fields": []},
            {"task_id": "report_generation", "decision": "retain", "supporting_fields": []},
        ],
        "question_designs": [
            {
                "task_id": "detection", "question_templates": ["Locate {target_text}."],
                "placeholder_bindings": {"target_text": "annotation.findings[*].kind"},
                "validation_rules": [], "citations": [], "instance_path": "annotation.findings[*]",
                "target_text_field": "annotation.findings[*].kind",
                "bbox_field": "annotation.findings[*].bbox[*]", "bbox_format": "xyxy",
            },
            {
                "task_id": "report_generation", "question_templates": ["Generate a report."],
                "placeholder_bindings": {}, "validation_rules": [], "citations": [],
                "answer_field": "annotation.report",
            },
        ],
    }

    result = workflow._generation_request_node(state)

    assert [spec["response_format"] for spec in result["generation_request"]["task_specs"]] == [
        "bbox",
        "free_text",
    ]
