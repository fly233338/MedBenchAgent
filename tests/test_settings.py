import json
import os
from pathlib import Path

import pytest

from medbenchagent.runtime.run_artifacts import initialize_run
from medbenchagent.settings import load_project_env, load_settings


def test_load_project_env_preserves_process_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("DEEPSEEK_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr("medbenchagent.settings.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-process")

    load_project_env()

    assert os.environ["DEEPSEEK_MODEL"] == "from-process"


def test_explicit_toml_then_environment_override_and_empty_value(tmp_path: Path, monkeypatch) -> None:
    local = tmp_path / "local.toml"
    local.write_text(
        "[llm]\nmodel = 'from-toml'\ntimeout_seconds = 30\n"
        "[rag]\nmin_rerank_score = 0.25\n"
        "[generation]\nquestion_template_count = 5\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-environment")
    monkeypatch.setenv("RAG_MIN_RERANK_SCORE", "")

    settings = load_settings(local)

    assert settings.llm.model == "from-environment"
    assert settings.llm.timeout_seconds == 30
    assert settings.rag.min_rerank_score == 0.25
    assert settings.generation.question_template_count == 5
    assert settings.environment_overrides == ("llm.model",)
    assert settings.local_config_path == local.resolve()


def test_explicit_toml_rejects_unknown_keys(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text("[generation]\nunknown = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown configuration key: generation.unknown"):
        load_settings(local)


def test_initialize_run_writes_secret_free_effective_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-in-snapshot")
    settings = load_settings()

    run = initialize_run(
        tmp_path / "runs",
        settings=settings,
        graph_overrides={"seed": 9, "question_temperature": 0.2},
    )

    snapshot = json.loads(Path(run["effective_config_path"]).read_text(encoding="utf-8"))
    assert snapshot["graph_overrides"] == {"seed": 9, "question_temperature": 0.2}
    assert snapshot["settings"]["generation"]["seed"] == 9
    assert snapshot["settings"]["generation"]["question_temperature"] == 0.2
    assert "not-in-snapshot" not in json.dumps(snapshot)
