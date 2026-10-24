from runtime_fixtures import runtime_for

import json
from pathlib import Path

import pytest

from medbenchagent.runtime.intake import (
    InputDraft,
    SelectedFile,
    suggest_draft,
    prepare_input,
    connection_values,
    save_connection,
)
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.sessions import create_session
from medbenchagent.data.task_data import SourceManifest


def test_directory_draft_creates_run_without_user_manifest(tmp_path, monkeypatch):
    import medbenchagent.runtime.model_services as services

    monkeypatch.setattr(services, "resolve_model_service", lambda *_: "offline")
    raw = tmp_path / "data with spaces"
    raw.mkdir()
    (raw / "labels.csv").write_text("id,label\n1,A\n", encoding="utf-8")
    draft = suggest_draft(f'用 "{raw}" 构建 4 道题')
    assert draft.source_dirs == [str(raw)] and draft.sample_count == 4
    draft.key_files = [SelectedFile(path=str(raw / "labels.csv"), purpose="标签表")]
    root = tmp_path / "experiments"
    create_session("default", root=root, runtime_root=runtime_for(root))
    service = ConversationService(root / "default")
    row = service.submit_draft(draft)
    options = json.loads((Path(row["directory"]) / "run.json").read_text(encoding="utf-8"))
    source = SourceManifest.model_validate_json(Path(options["source_manifest"]).read_bytes())
    assert source.key_files[0].path == "labels.csv" and source.key_files[0].purpose == "标签表"
    assert options["benchmark_request"]["sample_count"] == 4
    assert service.record()["draft"]["goal"] == draft.goal
    assert "status" not in service.record() and "runs" not in service.record()


def test_draft_boundaries_and_nonsecret_connection_settings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    draft = InputDraft(goal="Inspect", source_dirs=[str(raw)], goal_kind="profile")
    with pytest.raises(ValueError, match="contains"):
        prepare_input(draft, raw / "run")
    draft.key_files = [SelectedFile(path=str(tmp_path / "elsewhere.csv"))]
    with pytest.raises(ValueError, match="outside selected sources"):
        prepare_input(draft, tmp_path / "run")
    draft.goal = ""
    with pytest.raises(ValueError, match="Goal required"):
        prepare_input(draft, tmp_path / "run")
    create_session("settings", root=tmp_path / "runs", runtime_root=runtime_for(tmp_path / "runs"))
    config = tmp_path / "runs/settings/config.toml"
    values = connection_values(config)
    values.update(model="offline", api_key="never-store-this")
    save_connection(config, values)
    assert "never-store-this" not in config.read_text(encoding="utf-8")
    assert connection_values(config)["model"] == "offline"


def test_connection_probe_exposes_model_choices_and_skips_unused_coding(tmp_path, monkeypatch):
    import httpx

    import medbenchagent.runtime.model_services as models
    from medbenchagent.runtime.intake import probe_connections, check_connections

    create_session("connect", root=tmp_path, runtime_root=runtime_for(tmp_path))
    path = tmp_path / "connect/config.toml"
    values = connection_values(path)
    values["worker_venv"] = str(tmp_path / "missing-coding")
    save_connection(path, values)
    monkeypatch.setattr(
        models.httpx,
        "get",
        lambda url, **_: httpx.Response(
            200,
            json={"data": [{"id": "one"}, {"id": "two"}]},
            request=httpx.Request("GET", url),
        ),
    )
    draft = InputDraft(goal="Plan", goal_kind="scenarios", canonical_manifest="selected.json")
    assert probe_connections(path, draft) == {"model": ["one", "two"]}
    with pytest.raises(models.ModelDiscoveryError) as error:
        check_connections(path, draft)
    assert error.value.models == ["one", "two"]
    values["model"] = "two"
    save_connection(path, values)
    assert check_connections(path, draft) == {"model": "two"}
    with pytest.raises(ValueError, match="Coding Python"):
        probe_connections(path, InputDraft(goal="Profile", goal_kind="profile"))
