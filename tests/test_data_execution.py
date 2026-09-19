import json
import subprocess

import pytest

from medbenchagent.data import execution
from medbenchagent.settings import get_settings


def test_auto_falls_back_but_pinned_docker_does_not(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(execution, "docker_identity", unavailable)
    settings = {**get_settings().data_processing, "backend": "auto"}
    runtime = execution.select_backend(settings)
    assert runtime["backend"] == "local" and runtime["selection_reason"]
    with pytest.raises(ValueError, match="Docker"):
        execution.select_backend({**settings, "backend": "docker"})
    assert execution.select_backend(settings, runtime) == runtime


def test_local_fixed_operation_script_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "get_settings", lambda: type("Settings", (), {
        "data_processing": {**get_settings().data_processing, "backend": "local"}})())
    source = tmp_path / "中文 source"
    source.mkdir()
    (source / "a.csv").write_text("id,value\n01,a\n02,b\n", encoding="utf-8")
    context = {"source_root": str(source), "bundle_root": str(tmp_path / "bundle")}
    output = tmp_path / "table"
    result = execution.run(context, output, request={"operation": "table", "path": "a.csv"})
    assert result
    assert execution.run(context, output, request={"operation": "table", "path": "a.csv"}) == result
    script = tmp_path / "script.py"
    script.write_text('import os\nfrom pathlib import Path\nPath(os.environ["MEDBENCH_WORK_DIR"], "annotation.jsonl").write_text("{}\\n")', encoding="utf-8")
    execution.run(context, tmp_path / "script out", script=str(script))
    runtime = json.loads((tmp_path / "bundle/execution_runtime.json").read_text())
    assert runtime["backend"] == "local"
    script.write_text('raise RuntimeError("bad data")', encoding="utf-8")
    with pytest.raises(ValueError, match="bad data"):
        execution.run(context, tmp_path / "failed", script=str(script))
    assert not (tmp_path / "failed-script.complete.json").exists()


def test_local_timeout_terminates_process(tmp_path):
    import os
    import sys
    with pytest.raises(ValueError, match="budget"):
        execution._local_run([sys.executable, "-c", "import time; time.sleep(30)"],
                             cwd=tmp_path, env=os.environ.copy(), timeout=0.1)


@pytest.mark.parametrize("failure", [ValueError("missing image"), subprocess.TimeoutExpired("docker", 30)])
def test_auto_records_probe_failure(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(execution, "docker_identity", fail)
    assert execution.select_backend({"backend": "auto"})["selection_reason"]
