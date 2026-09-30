from medbenchagent.workflow.graph import available_stages


def test_unified_supervisor_exposes_preparation_and_end_tools(tmp_path):
    tools = available_stages({"run_directory": str(tmp_path), "user_requirement": "prepare a dataset"})
    assert "prepare_data" in tools
    assert "end" in tools
    assert "specify" in tools
