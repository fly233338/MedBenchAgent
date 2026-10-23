import json
from pathlib import Path
import pytest

from medbenchagent.runtime.intake import InputDraft, SelectedFile, suggest_draft, prepare_input, connection_values, save_connection
from medbenchagent.runtime.conversation_service import ConversationService
from medbenchagent.runtime.sessions import create_experiment, config_path
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
    create_experiment("default", root=root)
    service = ConversationService("default", root=root)
    row = service.submit_draft(draft)
    options = json.loads((Path(row["directory"]) / "run.json").read_text(encoding="utf-8"))
    source = SourceManifest.model_validate_json(Path(options["source_manifest"]).read_bytes())
    assert source.key_files[0].path == "labels.csv" and source.key_files[0].purpose == "标签表"
    assert options["benchmark_request"]["sample_count"] == 4
    assert service.record()["draft"]["goal"] == draft.goal
    assert "status" not in service.record()["runs"][0]


def test_draft_boundaries_and_nonsecret_connection_settings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    draft = InputDraft(goal="Inspect", source_dirs=[str(raw)], goal_kind="profile")
    with pytest.raises(ValueError, match="包含"):
        prepare_input(draft, raw / "run")
    draft.key_files = [SelectedFile(path=str(tmp_path / "elsewhere.csv"))]
    with pytest.raises(ValueError, match="来源目录"):
        prepare_input(draft, tmp_path / "run")
    draft.goal = ""
    with pytest.raises(ValueError, match="任务要求"):
        prepare_input(draft, tmp_path / "run")
    create_experiment("settings", root=tmp_path / "experiments")
    config = config_path("settings", tmp_path / "experiments")
    values = connection_values(config)
    values.update(model="offline", api_key="never-store-this")
    save_connection(config, values)
    assert "never-store-this" not in config.read_text(encoding="utf-8")
    assert connection_values(config)["model"] == "offline"
