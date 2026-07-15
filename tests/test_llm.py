import json
from pathlib import Path
from types import SimpleNamespace

import medbenchagent.llm as llm


def test_load_project_env_preserves_process_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("DEEPSEEK_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr(llm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-process")

    llm.load_project_env()

    assert llm.os.environ["DEEPSEEK_MODEL"] == "from-process"


def test_complete_json_uses_placeholder_key_for_unauthenticated_vllm(monkeypatch) -> None:
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
                    )
                )
            )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)

    assert llm.complete_json("system", "user") == {}
    assert captured["api_key"] == "local-vllm"


def test_langgraph_server_loads_project_env_file() -> None:
    config_path = Path(__file__).parents[1] / "langgraph.json"

    assert json.loads(config_path.read_text(encoding="utf-8"))["env"] == "./.env"
