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
        prompt = Path(request.prompt_path).read_text(encoding="utf-8")
        assert "Do not run preparation.cli validate" in prompt
        assert "Run sample and full checks" not in prompt
        assert "Use the shared validator" not in prompt
        assert "final completion action" not in prompt


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


@pytest.mark.parametrize("outcome", ["needs_input", "blocked"])
def test_signal_records_clarification_and_blocking_without_candidate_outputs(tmp_path, monkeypatch, outcome):
    import json
    from medbenchagent.preparation.cli import main
    from medbenchagent.preparation.contracts import StageControl

    monkeypatch.chdir(tmp_path)
    Path("request.json").write_text('{"stage":"understand"}', encoding="utf-8")
    arguments = ["signal", "--outcome", outcome, "--summary", "Need source meaning"]
    if outcome == "needs_input":
        questions = [{"issue_id": "meaning", "stage": "understand", "question": "Which label?", "impact": "mapping"}]
        Path("questions.json").write_text(json.dumps(questions), encoding="utf-8")
        arguments += ["--questions-file", "questions.json"]
    else:
        arguments += ["--diagnostic-code", "mapping_revision_required"]
    assert main(arguments) == 0
    signal = StageControl.model_validate_json(Path("work/control.json").read_text(encoding="utf-8"))
    assert signal.requested_outcome == outcome
    assert bool(signal.questions) == (outcome == "needs_input")
    assert not Path("work/outputs").exists()


def test_context_budget_fails_without_silent_truncation(tmp_path):
    context = state(tmp_path)
    context["user_request"] = "request" * 1000
    context["preparation_stage_budgets"] = {"understand": {"initial_message_tokens":256}}
    with pytest.raises(ContextBudgetError):
        build_stage_context(Stage.UNDERSTAND, context, tmp_path / "attempt")
