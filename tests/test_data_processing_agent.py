import pytest

from medbenchagent.agents.data_processing import finish, preparation_context, write_script


def test_invalid_script_is_a_repairable_error(tmp_path):
    with pytest.raises(ValueError, match="Invalid Python"):
        write_script(
            {"data_workspace": str(tmp_path)}, code="if :", mapping="Source mapping"
        )


def test_execution_alone_cannot_publish_data():
    with pytest.raises(ValueError, match="validate"):
        finish({"script_executed": True, "data_validated": False})


def test_reworking_completed_data_uses_a_new_workspace(tmp_path):
    context = {
        "source_root": "/source",
        "dataset_id": "set",
        "data_ready": True,
        "data_workspace": "/old/prepared",
        "operation_directory": str(tmp_path),
        "script_ready": True,
        "script_executed": True,
        "data_validated": True,
    }
    updated = preparation_context(context, "/source", "set")
    assert updated["data_workspace"] == str(tmp_path / "prepared")
    assert not updated["data_validated"]
    assert not updated["script_ready"]
