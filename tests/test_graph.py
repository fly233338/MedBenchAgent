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
                "answer_list": ["BI-RADS 2", "BI-RADS 4"],
                "answer_field": "annotation.breast_birads",
                "answer_value_mapping": {},
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
