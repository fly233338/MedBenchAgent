from medbenchagent.agents.task_space_pipeline import run_task_space_pipeline


def test_run_context_adapter_requires_persisted_artifacts(tmp_path):
    context = {"operation_directory": str(tmp_path), "dataset_access_manifest": {"schema_version": "annotation"}}
    try:
        run_task_space_pipeline(context)
    except ValueError as exc:
        assert "Axis Analysis" in str(exc)
    else:
        raise AssertionError("missing artifacts were accepted")

