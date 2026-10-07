from pathlib import Path
import json
import pytest

from medbenchagent.runtime.openhands_worker import normalize_tool_arguments, validate_terminal_command
from medbenchagent.preparation.contracts import Stage, StageRequest, StageResult
from medbenchagent.runtime.dataset_runtime import OpenHandsDatasetRuntime
from medbenchagent.preparation.worker import OpenHandsWorker


@pytest.mark.parametrize("kind", ["complete", "missing", "blocked", "needs_input", "invalid"])
def test_runtime_uses_outputs_by_default_and_exceptional_control_takes_precedence(tmp_path, kind):
    from medbenchagent.runtime.openhands_worker import candidate_outcome

    outputs = tmp_path / "work/outputs"
    outputs.mkdir(parents=True)
    if kind != "missing":
        (outputs / "mapping.json").write_text("{}", encoding="utf-8")
    if kind in {"blocked", "needs_input", "invalid"}:
        signal = {"requested_outcome": "submitted" if kind == "invalid" else kind}
        if kind == "needs_input":
            signal["questions"] = [{"issue_id": "meaning", "stage": "understand", "question": "Which label?", "impact": "mapping"}]
        (tmp_path / "work/control.json").write_text(json.dumps(signal), encoding="utf-8")
    outcome = candidate_outcome({"required_outputs": ["mapping.json"]}, tmp_path)
    expected = {"complete": "submitted", "missing": "failed", "invalid": "failed"}.get(kind, kind)
    assert outcome["status"] == expected
    assert not (tmp_path / "work/submission.json").exists()
    if kind == "complete":
        assert outcome["control_path"] is None


@pytest.mark.parametrize("kind", ["valid", "invalid", "missing"])
def test_host_validates_candidates_without_success_receipt_and_retries(tmp_path, kind):
    from test_preparation_graph import Worker, state
    from medbenchagent.preparation.graph import compile_preparation_graph
    from medbenchagent.runtime.openhands_worker import candidate_outcome

    class CandidateRuntime:
        def __init__(self):
            self.worker = Worker()

        def run(self, request):
            result = self.worker.run(request)
            root = Path(request.workspace)
            if request.stage is Stage.UNDERSTAND and request.attempt == 1:
                if kind == "invalid":
                    (root / "work/outputs/mapping.json").write_text("{}", encoding="utf-8")
                elif kind == "missing":
                    (root / "work/outputs/mapping.json").unlink()
            outcome = candidate_outcome(request.model_dump(mode="json"), root)
            return result.model_copy(update=outcome)

    runtime = CandidateRuntime()
    output = compile_preparation_graph(worker=OpenHandsWorker(runtime=runtime)).invoke(state(tmp_path))
    assert output["status"] == "complete", output.get("diagnostics")
    assert len(runtime.worker.requests) == (2 if kind == "valid" else 3)
    for request in runtime.worker.requests:
        assert not (Path(request.workspace) / "work/control.json").exists()
        prompt = Path(request.prompt_path).read_text(encoding="utf-8")
        assert "No success receipt or submit command is required" in prompt


def test_worker_preserves_runtime_failure_even_when_outputs_exist(tmp_path):
    output = tmp_path / "work/outputs/mapping.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    request = StageRequest(stage=Stage.UNDERSTAND, run_id="r", dataset_id="d", attempt=1,
                           conversation_id="c", execution_key="e", workspace=str(tmp_path),
                           prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "request.json"),
                           input_digest="input", required_outputs=["mapping.json"])

    class FailedRuntime:
        def run(self, request):
            return StageResult(stage=request.stage, attempt=1, conversation_id="c", execution_key="e",
                               status="failed", outputs={"mapping.json": str(output)}, termination_reason="worker_error")

    result = OpenHandsWorker(runtime=FailedRuntime()).run(request)
    assert result.status == "failed"
    assert result.termination_reason == "worker_error"


def test_tool_metadata_is_removed_without_touching_command():
    args, stripped = normalize_tool_arguments("terminal", {"command": "python script.py", "description_note": "why", "description": ""})
    assert args == {"command": "python script.py"}
    assert stripped == ["description_note", "description"]


def test_worker_rejects_dependency_install_and_network_commands():
    assert validate_terminal_command("python -m pip install scipy")
    assert validate_terminal_command("curl https://example.invalid")
    assert validate_terminal_command("python adapter.py") == ""


def test_runtime_validates_stage_result_paths_and_required_outputs(tmp_path):
    workspace = tmp_path / "attempt"
    output = workspace / "work" / "outputs" / "mapping.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(workspace),
        prompt_path=str(workspace / "prompt.md"), context_path=str(workspace / "work" / "inputs" / "context.json"),
        input_digest="input", required_outputs=["mapping.json"],
    )
    result = StageResult(
        stage=Stage.UNDERSTAND, attempt=1, conversation_id="conversation", execution_key="execution",
        status="submitted", outputs={"mapping.json": str(output)}, input_digest="input",
    )

    validated = OpenHandsDatasetRuntime._validate_outputs(request, result, workspace, "conversation", "local")

    assert validated.outputs["mapping.json"] == str(output.resolve())


def test_runtime_rejects_missing_worker_llm_configuration_before_launch(tmp_path):
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"),
        context_path=str(tmp_path / "work" / "inputs" / "context.json"), input_digest="input",
        required_outputs=["mapping.json"], model="", base_url="",
    )

    result = OpenHandsDatasetRuntime(runtime={"backend": "local"}).run(request)

    assert result.status == "blocked"
    assert result.termination_reason == "model_unavailable"
    assert "model is missing" in result.diagnostic


def test_runtime_preserves_structured_worker_failure(tmp_path):
    workspace = tmp_path / "attempt"
    result_path = workspace / "result.json"
    result_path.parent.mkdir(parents=True)
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(workspace),
        prompt_path=str(workspace / "prompt.md"),
        context_path=str(workspace / "work" / "inputs" / "context.json"), input_digest="input",
        required_outputs=["mapping.json"],
    )
    result_path.write_text(StageResult(
        stage=Stage.UNDERSTAND, attempt=1, conversation_id="conversation", execution_key="execution",
        status="failed", diagnostic="Preparation stage guard: no_progress",
        termination_reason="no_progress", input_digest="input",
    ).model_dump_json(), encoding="utf-8")

    result = OpenHandsDatasetRuntime()._result_after_process(
        request, workspace, result_path, "conversation", "local",
    )

    assert result.termination_reason == "no_progress"
    assert result.diagnostic == "Preparation stage guard: no_progress"


class _ControlRuntime:
    def __init__(self, stage):
        self.stage = stage

    def run(self, request):
        output = Path(request.workspace) / "work" / "control.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '{"requested_outcome":"blocked","summary":"accepted semantics conflict",'
            '"diagnostic_codes":["mapping_revision_required"]}',
            encoding="utf-8",
        )
        return StageResult(
            stage=request.stage, attempt=request.attempt, attempt_id=request.attempt_id,
            conversation_id=request.conversation_id, execution_key=request.execution_key,
            status="blocked", outputs={},
            input_digest=request.input_digest,
        )


def test_worker_projects_exceptional_control_for_prepare(tmp_path):
    request = StageRequest(
        stage=Stage.PREPARE_TEXT, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "context.json"),
        input_digest="input", required_outputs=["mapping.json"],
    )

    result = OpenHandsWorker(runtime=_ControlRuntime(Stage.PREPARE_TEXT)).run(request)

    assert result.control == "mapping_revision_required"
    assert result.status == "blocked"
    assert any(item.code == "mapping_revision_required" for item in result.diagnostics)


def test_worker_rejects_mapping_control_for_understand(tmp_path):
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "context.json"),
        input_digest="input", required_outputs=["mapping.json"],
    )

    result = OpenHandsWorker(runtime=_ControlRuntime(Stage.UNDERSTAND)).run(request)

    assert result.control is None
    assert result.status == "blocked"
    assert any(item.code == "invalid_control_stage" for item in result.diagnostics)


def test_local_worker_command_uses_frozen_interpreter(tmp_path):
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "context.json"),
        input_digest="input", required_outputs=["mapping.json"],
        worker_runtime={"python": str(tmp_path / "python.exe")},
    )

    command = OpenHandsDatasetRuntime._local_command(
        request, tmp_path / "worker.py", tmp_path / "request.json", tmp_path / "result.json",
    )

    assert command == [str(tmp_path / "python.exe"), str(tmp_path / "worker.py"),
                      str(tmp_path / "request.json"), str(tmp_path / "result.json")]
    assert "uv" not in command
