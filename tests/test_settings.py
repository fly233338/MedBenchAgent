from pathlib import Path
import json
import os
import pytest
from medbenchagent.settings import load_project_env

def test_compilation_coding_config_overrides_common_llm_and_freezes_without_credentials(tmp_path, monkeypatch):
    import medbenchagent.settings as settings

    monkeypatch.setattr(settings, "load_project_env", lambda: None)
    for name in (*settings._ENVIRONMENT_OVERRIDES, "MEDBENCH_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "common-environment-model")
    monkeypatch.setenv("MEDBENCH_LLM_BASE_URL", "http://common.invalid/v1")
    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "shared-secret")
    monkeypatch.setenv("MEDBENCH_CODING_LLM_API_KEY", "coding-secret")
    local = tmp_path / "coding.toml"
    local.write_text(
        '[llm]\nmodel="common-file-model"\ntimeout_seconds=20\n'
        '[coding]\nworker_venv="coding-env"\nmax_iterations=7\n'
        '[coding.llm]\nmodel="independent-model"\nreasoning_effort="low"\n'
        '[rag]\nchunk_size="must not be parsed"\n'
        '[data_processing]\nbackend="must not be initialized"\n', encoding="utf-8",
    )
    monkeypatch.setattr(settings, "load_settings", lambda *_: pytest.fail("Do not load full app settings"))
    supervisor = settings.load_supervisor_settings(local)
    coding = settings.load_coding_settings(local)
    assert supervisor.llm.model == "common-environment-model"
    assert coding.llm.model == "independent-model" and coding.llm.reasoning_effort == "low"
    assert coding.llm.base_url == supervisor.llm.base_url == "http://common.invalid/v1"
    assert coding.llm.timeout_seconds == 20
    assert Path(coding.worker_python).parent.parent == (settings.PROJECT_ROOT / "coding-env").resolve()
    assert coding.max_iterations == 7
    assert settings.get_coding_api_key() == "coding-secret"
    snapshot = coding.snapshot()
    assert "secret" not in json.dumps(snapshot)
    local.write_text('[coding.llm]\nmodel="changed"\n', encoding="utf-8")
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "changed-env")
    assert settings.CodingSettings.from_snapshot(snapshot) == coding
    monkeypatch.delenv("MEDBENCH_CODING_LLM_API_KEY")
    assert settings.get_coding_api_key() == "shared-secret"
    local.write_text('[coding.llm]\napi_key="must-not-be-accepted"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key: coding.llm.api_key"):
        settings.load_coding_settings(local)

def test_load_project_env_preserves_process_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("MEDBENCH_LLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr("medbenchagent.settings.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "from-process")

    load_project_env()

    assert os.environ["MEDBENCH_LLM_MODEL"] == "from-process"
