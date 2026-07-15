from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import medbenchagent.workflow.graph as workflow


def test_graph_reaches_generation_request_with_confirmations(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "cases.csv").write_text(
        "case_id,image_path,diagnosis\n1,image-1.png,benign\n2,image-2.png,malignant\n", encoding="utf-8"
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
    monkeypatch.setattr(workflow, "propose_schema", lambda _inventory, schema: schema)

    class FakeRag:
        def __init__(self, *_args):
            pass

    monkeypatch.setattr(workflow, "LocalPdfRag", FakeRag)
    monkeypatch.setattr(
        workflow,
        "design_questions",
        lambda _rag, _info, plans, _feedback: [{"task_id": plan["task_id"], "citations": []} for plan in plans],
    )
    graph = workflow.compile_benchmark_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread"}}
    result = graph.invoke({"messages": [HumanMessage(content="make a benchmark")]}, config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume="yes"), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume=str(dataset_root)), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume="yes"), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume=str(tmp_path / "materials")), config)
    assert result["__interrupt__"]
    result = graph.invoke(Command(resume="yes"), config)
    assert result["ready_for_generation"] is True
    assert result["generation_request"]["target_sample_count"] == 2
