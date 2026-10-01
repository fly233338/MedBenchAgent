from medbenchagent.agents.v2_pipeline import run_task_space_pipeline


def test_run_context_adapter_requires_persisted_artifacts(tmp_path):
    context = {"operation_directory": str(tmp_path), "dataset_access_manifest": {"schema_version": "annotation.v5"}}
    try:
        run_task_space_pipeline(context)
    except ValueError as exc:
        assert "Axis Analysis" in str(exc)
    else:
        raise AssertionError("missing v2 artifacts were accepted")

