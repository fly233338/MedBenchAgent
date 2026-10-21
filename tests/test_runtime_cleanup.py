import json
from dataclasses import replace
from pathlib import Path

import pytest

from test_preparation_graph import Worker, state
from medbenchagent.preparation.contracts import Stage, ValidationResult, Diagnostic
from medbenchagent.preparation.graph import _retry, compile_preparation_graph
from medbenchagent.settings import active_settings, get_settings, load_settings
from medbenchagent.workflow.graph import child_result


def test_repair_progress_requires_new_candidate_and_host_improvement(tmp_path):
    root = tmp_path / "work/outputs"
    root.mkdir(parents=True)
    candidate = root / "mapping.json"
    current = {"attempt_workspace": str(tmp_path)}
    issue = Diagnostic(stage=Stage.UNDERSTAND, code="invalid_output", message="fix the mapping")
    check = ValidationResult(status="rejected", stage=Stage.UNDERSTAND, issues=[issue], passed_checks=["outputs"])

    def retry(content, checks):
        candidate.write_text(content, encoding="utf-8")
        current.update(_retry(current, Stage.UNDERSTAND, [issue.model_dump(mode="json")],
                              check.model_copy(update={"passed_checks": checks})))

    retry("first", ["outputs"])
    assert current["understand_failures"] == 0
    retry("changed but still invalid", ["outputs"])
    assert current["understand_failures"] == 1
    retry("fixed schema", ["outputs", "schema"])
    assert current["understand_failures"] == 0
    assert current["validation_feedback"]["advanced"]
    # A restored checkpoint preserves the best validation result and budget.
    current = json.loads(json.dumps(current))
    retry("regression", ["outputs"])
    retry("bounce to already achieved result", ["outputs", "schema"])
    assert current["status"] == "blocked"
    assert current["understand_failures"] == 2


class RepairingWorker(Worker):
    def run(self, request):
        result = super().run(request)
        if request.stage is Stage.UNDERSTAND and request.attempt <= 2:
            path = Path(result.outputs["mapping.json"])
            value = json.loads(path.read_text(encoding="utf-8"))
            if request.attempt == 1:
                value = {}
            else:
                value["image_rules"][0]["source_id"] = "unknown"
            path.write_text(json.dumps(value), encoding="utf-8")
        return result


def test_progressing_repairs_can_exceed_no_progress_limit(tmp_path):
    initial = state(tmp_path)
    initial["preparation_attempt_limits"] = {"understand": 1, "prepare_text": 1, "prepare_image": 1, "mapping_revision_cycles": 0}
    worker = RepairingWorker()
    output = compile_preparation_graph(worker=worker).invoke(initial)
    assert output["status"] == "complete"
    assert output["understand_attempts"] == 3
    assert output["understand_failures"] == 0
    assert len({request.conversation_id for request in worker.requests if request.stage is Stage.UNDERSTAND}) == 1


def test_stage_hard_budget_stops_even_progressing_repairs(tmp_path):
    initial = state(tmp_path)
    initial["preparation_stage_budgets"] = {"understand": {"max_iterations": 2}}
    worker = RepairingWorker()
    output = compile_preparation_graph(worker=worker).invoke(initial)
    assert output["status"] == "blocked"
    assert output["diagnostics"][0]["code"] == "stage_budget_exhausted"
    assert [request.budget.max_iterations for request in worker.requests] == [2, 1]
    # Re-entering the graph must not replenish execution allowance.
    resumed = compile_preparation_graph(worker=worker).invoke(output)
    assert resumed["status"] == "blocked" and len(worker.requests) == 2


@pytest.mark.parametrize("status", ["complete", "progress", "needs_input", "failed", "blocked"])
def test_child_business_outcome_is_distinct_from_delegation_success(tmp_path, status):
    child = {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "child"), "dataset_scope": "d"}
    outcome = child_result({}, "d", "planning", {"context": child, "result": {
        "status": status, "summary": "child result", "diagnostics": {"reason": "detail"}}})
    assert outcome["status"] == "success"
    assert outcome["agent_outcome"]["status"] == status
    assert outcome["agent_outcome"]["diagnostics"] == {"reason": "detail"}
    assert Path(outcome["agent_outcome"]["artifacts"]["private_state"]).is_file()
    assert "agent_outcome" not in outcome["updates"]


@pytest.mark.parametrize("section,key", [
    ("planning", "novelty_enabled"), ("generation", "question_temperature"),
    ("data_processing", "source_workspace_mode"), ("data_processing", "coding_worker_max_iterations"),
    ("data_processing", "coding_worker_timeout_seconds"),
    ("data_processing", "image"), ("data_processing", "timeout_seconds"),
    ("data_processing", "memory_limit"), ("data_processing", "threads"), ("data_processing", "container_memory"),
])
def test_removed_configuration_is_rejected(tmp_path, section, key):
    path = tmp_path / "config.toml"
    path.write_text(f"[{section}]\n{key} = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_settings(path)


def test_generation_fetch_batch_size_controls_real_pagination(tmp_path, monkeypatch):
    from compilation_fixtures import compile_fixture
    from test_compilation import cases
    from test_global_construction import _plan
    from medbenchagent.agents.generation import run_generation
    from medbenchagent.data.index import DatasetIndex
    records = [{**cases()[0], "case_id": str(index)} for index in range(3)]
    context = compile_fixture(tmp_path, _plan("d", {"leaf": 3}), records, requested=3)
    reads = []
    original = DatasetIndex.sample
    def sample(self, **kwargs):
        reads.append((kwargs["offset"], kwargs["limit"]))
        return original(self, **kwargs)
    monkeypatch.setattr(DatasetIndex, "sample", sample)
    settings = get_settings()
    with active_settings(replace(settings, generation=replace(settings.generation, fetch_batch_size=1))):
        run_generation(context, output_path=tmp_path / "paged.jsonl")
    assert reads == [(0, 1), (1, 1), (2, 1)]


def test_excel_reader_preserves_named_sheet_accounting(tmp_path):
    import pandas as pd
    from medbenchagent.data.tables import read_excel
    path = tmp_path / "sources.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"id": ["unused"]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"id": ["a", "b"]}).to_excel(writer, sheet_name="records", index=False)
    assert read_excel(path)["id"].tolist() == ["unused"]
    assert read_excel(path, sheet="records")["id"].tolist() == ["a", "b"]


def test_configuration_example_is_loadable():
    assert load_settings(Path("config/local.toml.example")).llm.reasoning_effort == "medium"


def test_fixing_one_schema_error_counts_as_host_verified_progress(tmp_path):
    from test_preparation_graph import mapping
    from medbenchagent.preparation.validation import validate_stage
    initial = state(tmp_path)
    initial["attempt_workspace"] = str(tmp_path / "attempt")
    root = tmp_path / "attempt/work/outputs"
    root.mkdir(parents=True)
    candidate = root / "mapping.json"
    candidate.write_text("{}", encoding="utf-8")
    (root / "image_conversion_plan.json").write_text('{"policies":{}}', encoding="utf-8")
    first = validate_stage(root, Stage.UNDERSTAND, initial)
    initial.update(_retry(initial, Stage.UNDERSTAND, [item.model_dump() for item in first.issues], first))
    candidate.write_text(json.dumps({"population_policy": mapping()["population_policy"]}), encoding="utf-8")
    improved = validate_stage(root, Stage.UNDERSTAND, initial)
    assert improved.passed_checks == first.passed_checks and len(improved.issues) < len(first.issues)
    update = _retry(initial, Stage.UNDERSTAND, [item.model_dump() for item in improved.issues], improved)
    assert update["validation_feedback"]["advanced"] and update["understand_failures"] == 0


def test_restored_stage_time_budget_blocks_without_starting_worker(tmp_path):
    initial = state(tmp_path)
    initial["stage_usage"] = {"understand": {"iterations": 1, "seconds": 1800.0}}
    worker = Worker()
    output = compile_preparation_graph(worker=worker).invoke(initial)
    assert output["status"] == "blocked" and not worker.requests
    assert output["diagnostics"][0]["code"] == "stage_budget_exhausted"


def test_agent_tool_loop_checks_final_request_before_model_call(tmp_path, monkeypatch):
    from medbenchagent.runtime import llm
    from medbenchagent.runtime.harness import compile_agent, result
    monkeypatch.setattr(llm, "resolve_model_service", lambda *_args: "test-model")
    monkeypatch.setattr(llm, "_client", lambda: pytest.fail("Must not call the model"))
    graph = compile_agent("BudgetAgent", "required instructions", {}, lambda _context: {},
                          lambda _context: result("complete", "done"), request_char_budget=lambda _context: 1)
    output = graph.invoke({"context": {"run_id": "r", "run_directory": str(tmp_path),
        "operation_directory": str(tmp_path / "operations"), "construction_events_path": str(tmp_path / "events.jsonl")}})
    assert output["result"]["status"] == "blocked"
    assert output["result"]["diagnostics"]["request_chars"] > 1
