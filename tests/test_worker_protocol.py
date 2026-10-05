from medbenchagent.runtime.openhands_worker import normalize_tool_arguments, validate_terminal_command
from medbenchagent.preparation.contracts import Stage, StageRequest, StageResult
from medbenchagent.runtime.dataset_runtime import OpenHandsDatasetRuntime


def test_tool_metadata_is_removed_without_touching_command():
    args, stripped = normalize_tool_arguments("terminal", {"command": "python script.py", "description_note": "why"})
    assert args == {"command": "python script.py"}
    assert stripped == ["description_note"]


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
