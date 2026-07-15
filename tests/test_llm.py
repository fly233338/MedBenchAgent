from pathlib import Path

import medbenchagent.llm as llm


def test_load_project_env_preserves_process_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("DEEPSEEK_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr(llm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-process")

    llm.load_project_env()

    assert llm.os.environ["DEEPSEEK_MODEL"] == "from-process"
