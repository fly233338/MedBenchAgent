from medbenchagent.runtime.openhands_worker import normalize_tool_arguments, validate_terminal_command


def test_tool_metadata_is_removed_without_touching_command():
    args, stripped = normalize_tool_arguments("terminal", {"command": "python script.py", "description_note": "why"})
    assert args == {"command": "python script.py"}
    assert stripped == ["description_note"]


def test_worker_rejects_dependency_install_and_network_commands():
    assert validate_terminal_command("python -m pip install scipy")
    assert validate_terminal_command("curl https://example.invalid")
    assert validate_terminal_command("python adapter.py") == ""
