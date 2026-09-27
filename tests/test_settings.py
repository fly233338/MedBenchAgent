import json
import os
from pathlib import Path

import pytest

from medbenchagent.runtime.run_artifacts import initialize_run
from medbenchagent.settings import load_project_env, load_settings


@pytest.mark.parametrize("name,value,section,key,expected", [
    ("DEEPSEEK_TIMEOUT_SECONDS", "180", "llm", "timeout_seconds", 180),
    ("MEDBENCH_LLM_TIMEOUT_SECONDS", "0.5", "llm", "timeout_seconds", 0.5),
    ("MEDBENCH_LLM_MAX_RETRIES", "0", "llm", "max_retries", 0),
    ("RAG_MIN_RERANK_SCORE", "0.5", "rag", "min_rerank_score", 0.5),
    ("RAG_MIN_RERANK_SCORE", "1", "rag", "min_rerank_score", 1),
])
def test_numeric_environment_overrides(monkeypatch, name, value, section, key, expected):
    monkeypatch.setenv(name, value)
    assert getattr(getattr(load_settings(), section), key) == expected


@pytest.mark.parametrize("name,value", [
    ("MEDBENCH_LLM_TIMEOUT_SECONDS", "zero"), ("MEDBENCH_LLM_TIMEOUT_SECONDS", "0"),
    ("MEDBENCH_LLM_TIMEOUT_SECONDS", "inf"), ("RAG_MIN_RERANK_SCORE", "nan"),
    ("RAG_MIN_RERANK_SCORE", "1.1"), ("MEDBENCH_LLM_MAX_RETRIES", "1.5"),
    ("MEDBENCH_LLM_MAX_RETRIES", "-1"),
])
def test_invalid_numeric_environment_names_variable(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_settings()


def test_neutral_names_take_precedence_and_empty_names_use_alias(monkeypatch):
    from medbenchagent.settings import get_llm_api_key
    for suffix in ("MODEL", "API_KEY"):
        monkeypatch.setenv("DEEPSEEK_" + suffix, "legacy")
        monkeypatch.setenv("MEDBENCH_LLM_" + suffix, "new")
    assert load_settings().llm.model == "new"
    assert get_llm_api_key() == "new"
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "")
    monkeypatch.setenv("MEDBENCH_LLM_API_KEY", "")
    assert load_settings().llm.model == "legacy"
    assert get_llm_api_key() is None


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


def test_cli_config_path_selects_file_over_environment_and_snapshots_remote_rag(tmp_path, monkeypatch):
    from medbenchagent.settings import active_settings, get_settings, run_settings

    selected = tmp_path / "selected.toml"
    selected.write_text("[llm]\nmodel='selected'\n"
                        "[rag]\nembedding_base_url='http://embed/v1'\n"
                        "embedding_model='embed-v1'\nreranker_base_url='http://rerank/v1'\n"
                        "reranker_model='rerank-v1'\n", encoding="utf-8")
    from_env = tmp_path / "from-env.toml"
    from_env.write_text("[llm]\nmodel='from-env'\n", encoding="utf-8")
    monkeypatch.setenv("MEDBENCH_CONFIG", str(from_env))
    monkeypatch.setenv("MEDBENCH_LLM_MODEL", "field-override")
    monkeypatch.chdir(tmp_path)

    assert load_settings().local_config_path == from_env
    settings = load_settings(Path("selected.toml"))
    assert settings.local_config_path == selected
    assert settings.llm.model == "field-override"
    assert settings.rag.embedding_model == "embed-v1"
    with active_settings(settings):
        assert get_settings() is settings
        run = initialize_run(tmp_path / "runs", settings=settings)
    selected.write_text("[rag]\nembedding_model='changed'\n", encoding="utf-8")
    with run_settings(run["effective_config_path"]) as saved:
        assert saved.rag.embedding_model == "embed-v1"
        assert saved.rag.reranker_base_url == "http://rerank/v1"
    snapshot = json.loads(Path(run["effective_config_path"]).read_text(encoding="utf-8"))
    assert snapshot["config_sources"]["local"]["path"] == str(selected)
    assert snapshot["environment_overrides"] == ["llm.model"]


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


def test_source_settings_normalize_paths_and_resume_snapshot(tmp_path, monkeypatch):
    from medbenchagent.settings import run_settings
    monkeypatch.setattr("medbenchagent.settings.PROJECT_ROOT", tmp_path)
    local = tmp_path / "sources.toml"
    local.write_text("[[dataset_sources]]\ndataset_id='ecd'\nname='ECD'\ndescription='Original dataset'\n"
                     "[[dataset_sources.sources]]\nid='images'\nroot='raw/ecd'\n", encoding="utf-8")
    cfg = load_settings(local)
    assert cfg.dataset_sources[0]["sources"][0]["root"] == str(tmp_path / "raw/ecd")
    run = initialize_run(tmp_path / "runs", settings=cfg)
    local.write_text("dataset_sources=[]\n", encoding="utf-8")
    with run_settings(run["effective_config_path"]) as saved:
        assert saved.dataset_sources == cfg.dataset_sources


@pytest.mark.parametrize("text,match", [
    ("[[dataset_sources]]\ndataset_id='a'\nname='A'\ndescription='Data'\nsources=[]\n", "at least 1"),
    ("[[dataset_sources]]\ndataset_id='A/b'\nname='A'\ndescription='Data'\nsources=[]\n", "unsafe id"),
    ("[[dataset_sources]]\ndataset_id='a'\nname='A'\ndescription='Data'\n"
     "sources=[{id='x',root='raw'}, {id='x',root='raw2'}]\n", "Source IDs must be unique"),
    (("[[dataset_sources]]\ndataset_id='a'\nname='A'\ndescription='Data'\nsources=[{id='x',root='raw'}]\n") * 2,
     "Duplicate dataset_sources"),
])
def test_invalid_source_configuration(tmp_path, text, match):
    local = tmp_path / "sources.toml"
    local.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_settings(local)


def test_benchmark_settings_are_saved_and_validated(tmp_path):
    from medbenchagent.settings import run_settings
    local = tmp_path / "defaults.toml"
    local.write_text("[benchmark]\ndefault_sample_count=35\n", encoding="utf-8")
    cfg = load_settings(local)
    run = initialize_run(tmp_path / "runs", settings=cfg)
    local.write_text("[benchmark]\ndefault_sample_count=1\n", encoding="utf-8")
    with run_settings(run["effective_config_path"]) as saved:
        assert saved.benchmark.default_sample_count == 35
    for section, key in (("benchmark", "default_sample_count"),):
        local.write_text(f"[{section}]\n{key}=0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="positive"):
            load_settings(local)


def test_removed_semantic_audit_config_is_rejected(tmp_path):
    local = tmp_path / "obsolete.toml"
    local.write_text("[audit]\nsemantic_samples_per_task=10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration key: audit"):
        load_settings(local)
