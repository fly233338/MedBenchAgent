from pathlib import Path

from medbenchagent.runtime.openhands_worker import normalize_tool_arguments, validate_terminal_command
from medbenchagent.preparation.contracts import Stage, StageRequest, StageResult
from medbenchagent.runtime.dataset_runtime import OpenHandsDatasetRuntime
from medbenchagent.preparation.worker import OpenHandsWorker


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
    output = workspace / "outputs" / "submission.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(workspace),
        prompt_path=str(workspace / "inputs" / "prompt.md"), context_path=str(workspace / "inputs" / "context.json"),
        input_digest="input", required_outputs=["submission.json"],
    )
    result = StageResult(
        stage=Stage.UNDERSTAND, attempt=1, conversation_id="conversation", execution_key="execution",
        status="submitted", outputs={"submission.json": str(output)}, input_digest="input",
    )

    validated = OpenHandsDatasetRuntime._validate_outputs(request, result, workspace, "conversation", "local")

    assert validated.outputs["submission.json"] == str(output.resolve())


def test_runtime_rejects_missing_worker_llm_configuration_before_launch(tmp_path):
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "inputs" / "prompt.md"),
        context_path=str(tmp_path / "inputs" / "context.json"), input_digest="input",
        required_outputs=["submission.json"], model="", base_url="",
    )

    result = OpenHandsDatasetRuntime(runtime={"backend": "local"}).run(request)

    assert result.status == "blocked"
    assert result.termination_reason == "model_unavailable"
    assert "model is missing" in result.diagnostic


def test_runtime_preserves_structured_worker_failure(tmp_path):
    workspace = tmp_path / "attempt"
    result_path = workspace / "state" / "result.json"
    result_path.parent.mkdir(parents=True)
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(workspace),
        prompt_path=str(workspace / "inputs" / "prompt.md"),
        context_path=str(workspace / "inputs" / "context.json"), input_digest="input",
        required_outputs=["submission.json"],
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


class _SubmissionRuntime:
    def __init__(self, stage):
        self.stage = stage

    def run(self, request):
        output = Path(request.workspace) / "outputs" / "submission.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            '{"requested_outcome":"blocked","summary":"accepted semantics conflict",'
            '"diagnostic_codes":["mapping_revision_required"]}',
            encoding="utf-8",
        )
        return StageResult(
            stage=request.stage, attempt=request.attempt, attempt_id=request.attempt_id,
            conversation_id=request.conversation_id, execution_key=request.execution_key,
            status="blocked", outputs={"submission.json": str(output)},
            input_digest=request.input_digest,
        )


def test_worker_projects_submission_control_for_prepare(tmp_path):
    request = StageRequest(
        stage=Stage.PREPARE_TEXT, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "context.json"),
        input_digest="input", required_outputs=["submission.json"],
    )

    result = OpenHandsWorker(runtime=_SubmissionRuntime(Stage.PREPARE_TEXT)).run(request)

    assert result.control == "mapping_revision_required"
    assert result.status == "blocked"
    assert any(item.code == "mapping_revision_required" for item in result.diagnostics)


def test_worker_rejects_mapping_control_for_understand(tmp_path):
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "context.json"),
        input_digest="input", required_outputs=["submission.json"],
    )

    result = OpenHandsWorker(runtime=_SubmissionRuntime(Stage.UNDERSTAND)).run(request)

    assert result.control is None
    assert result.status == "blocked"
    assert any(item.code == "invalid_control_stage" for item in result.diagnostics)


def test_local_worker_command_uses_frozen_interpreter(tmp_path):
    request = StageRequest(
        stage=Stage.UNDERSTAND, run_id="run", dataset_id="dataset", attempt=1,
        conversation_id="conversation", execution_key="execution", workspace=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"), context_path=str(tmp_path / "context.json"),
        input_digest="input", required_outputs=["submission.json"],
        worker_runtime={"python": str(tmp_path / "python.exe")},
    )

    command = OpenHandsDatasetRuntime._local_command(
        request, tmp_path / "worker.py", tmp_path / "request.json", tmp_path / "result.json",
    )

    assert command == [str(tmp_path / "python.exe"), str(tmp_path / "worker.py"),
                      str(tmp_path / "request.json"), str(tmp_path / "result.json")]
    assert "uv" not in command
