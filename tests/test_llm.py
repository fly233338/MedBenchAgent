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
    instances = 0

    class FakeOpenAI:
        def __init__(self, **kwargs):
            nonlocal instances
            instances += 1
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
                    )
                )
            )

        def close(self) -> None:
            pass

    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    llm.clear_client_cache()

    assert llm.complete_json("system", "user") == {}
    assert llm.complete_json("system", "user") == {}
    assert captured["api_key"] == "local-vllm"
    assert captured["timeout"] == 120.0
    assert captured["max_retries"] == 0
    assert instances == 1
    llm.clear_client_cache()


def test_complete_json_rejects_invalid_timeout_before_request(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "0")
    llm.clear_client_cache()

    try:
        llm.complete_json("system", "user")
    except llm.ModelUnavailable as exc:
        assert "positive number" in str(exc)
    else:
        raise AssertionError("Invalid timeout should fail before a request.")


def test_langgraph_server_loads_project_env_file() -> None:
    config_path = Path(__file__).parents[1] / "langgraph.json"

    assert json.loads(config_path.read_text(encoding="utf-8"))["env"] == "./.env"
