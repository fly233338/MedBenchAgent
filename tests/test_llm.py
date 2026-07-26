import os
from pathlib import Path
from types import SimpleNamespace

import medbenchagent.llm as llm
from medbenchagent.settings import clear_settings_cache


def test_load_project_env_preserves_process_environment(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("DEEPSEEK_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setattr("medbenchagent.settings.PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("DEEPSEEK_MODEL", "from-process")

    llm.load_project_env()

    assert os.environ["DEEPSEEK_MODEL"] == "from-process"


def test_complete_json_uses_placeholder_key_for_unauthenticated_vllm(monkeypatch) -> None:
    captured = {}
    instances = 0
    wrapped_clients = []

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
    monkeypatch.setattr(
        llm,
        "wrap_openai",
        lambda client, **kwargs: wrapped_clients.append((client, kwargs)) or client,
    )
    clear_settings_cache()
    llm.clear_client_cache()

    assert llm.complete_json("system", "user") == {}
    assert llm.complete_json("system", "user") == {}
    assert captured["api_key"] == "local-vllm"
    assert captured["timeout"] == 120.0
    assert captured["max_retries"] == 0
    assert instances == 1
    assert len(wrapped_clients) == 1
    assert wrapped_clients[0][1] == {"chat_name": "DeepSeek"}
    llm.clear_client_cache()
    clear_settings_cache()
