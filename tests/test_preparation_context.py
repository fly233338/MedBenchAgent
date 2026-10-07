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
        assert request.context["source_manifests"] == request.source_manifests


def test_worker_cli_validates_using_frozen_inventory(tmp_path, monkeypatch):
    from medbenchagent.preparation.cli import main

    worker = Worker()
    result = compile_preparation_graph(worker=worker).invoke(state(tmp_path))
    assert result["status"] == "complete"
    for request in worker.requests:
        monkeypatch.chdir(request.workspace)
        assert main(["validate"]) == 0


def test_docker_context_exposes_mounted_inventory(tmp_path):
    context = state(tmp_path)
    context["execution_runtime"] = {"backend": "docker"}
    visible, _, _, _ = build_stage_context(Stage.UNDERSTAND, context, tmp_path / "attempt")
    assert visible["source_manifests"] == {"s": "/workspace/source_manifests/s"}


def test_context_budget_fails_without_silent_truncation(tmp_path):
    context = state(tmp_path)
    context["user_request"] = "request" * 1000
    context["preparation_stage_budgets"] = {"understand": {"initial_message_tokens":256}}
    with pytest.raises(ContextBudgetError):
        build_stage_context(Stage.UNDERSTAND, context, tmp_path / "attempt")
