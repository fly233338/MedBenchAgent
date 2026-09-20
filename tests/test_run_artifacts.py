import json
import re
from dataclasses import replace
from pathlib import Path

from medbenchagent.runtime.run_artifacts import initialize_run
from medbenchagent.settings import get_settings


def test_initialize_run_prepares_uuid_directory_and_event_file(tmp_path: Path) -> None:
    run = initialize_run(tmp_path / "runs")

    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", run["run_id"])
    assert Path(run["construction_events_path"]).is_file()
    assert Path(run["run_directory"]).is_dir()
    assert "working_blueprint_path" not in run
    event = json.loads(Path(run["construction_events_path"]).read_text(encoding="utf-8"))
    assert event["event_type"] == "run_started"
    assert event["component"] == "orchestrator"
    assert "agent" not in event


def test_run_freezes_blueprint_and_materials_but_not_catalog(tmp_path):
    template = tmp_path / "template.json"
    template.write_text('{"concepts": {}}', encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text('{"datasets": []}', encoding="utf-8")
    materials = tmp_path / "materials"
    materials.mkdir()
    (materials / "source.pdf").write_bytes(b"reference content")
    settings = get_settings()
    settings = replace(settings, paths=replace(settings.paths, blueprint_template=template,
                       dataset_catalog=catalog, materials_root=materials))
    run = initialize_run(tmp_path / "runs", settings=settings)
    paths = json.loads(Path(run["effective_config_path"]).read_text(encoding="utf-8"))["settings"]["paths"]
    assert paths["dataset_catalog"] == str(catalog)
    assert not (Path(run["run_directory"]) / ".runtime/inputs/catalog.json").exists()
    template.write_text("changed", encoding="utf-8")
    assert Path(paths["blueprint_template"]).read_text(encoding="utf-8") == '{"concepts": {}}'
    assert (Path(paths["materials_root"]) / "source.pdf").read_bytes() == b"reference content"
