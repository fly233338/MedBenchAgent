from pathlib import Path
import pytest
from test_preparation_graph import Worker, state
from medbenchagent.preparation.graph import compile_preparation_graph
from medbenchagent.preparation.context import build_stage_context, ContextBudgetError
from medbenchagent.preparation.contracts import Stage


def test_context_materializes_only_semantic_inputs(tmp_path):
    worker = Worker()
    result = compile_preparation_graph(worker=worker).invoke(state(tmp_path))
    assert result["status"] == "complete"
    expected = [set(), {"mapping.json", "image_conversion_plan.json"}, {"mapping.json", "image_conversion_plan.json", "cases.jsonl"}]
    for request, names in zip(worker.requests, expected):
        actual = {p.name for p in (Path(request.workspace) / "work/inputs").iterdir()}
        assert actual == names
        assert not (Path(request.workspace) / "work/inputs/context.json").exists()
        assert set(request.context["inputs"]) == names


def test_context_budget_fails_without_silent_truncation(tmp_path):
    context = state(tmp_path)
    context["user_request"] = "request" * 1000
    context["preparation_stage_budgets"] = {"understand": {"initial_message_tokens":256}}
    with pytest.raises(ContextBudgetError):
        build_stage_context(Stage.UNDERSTAND, context, tmp_path / "attempt")
