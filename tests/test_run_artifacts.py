import json
import re
from pathlib import Path

from medbenchagent.run_artifacts import initialize_run, update_blueprint_dataset


def test_initialize_run_prepares_uuid_directory_and_event_file(tmp_path: Path) -> None:
    run = initialize_run(tmp_path / "runs")

    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", run["run_id"])
    assert Path(run["construction_events_path"]).is_file()
    assert Path(run["working_blueprint_path"]).parent == tmp_path / "runs" / run["run_id"]


def test_dataset_snapshot_is_replaced_atomically(tmp_path: Path) -> None:
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps({"tasks": {}}), encoding="utf-8")

    update_blueprint_dataset(
        blueprint_path,
        {"dataset_id": "vindr-mammo", "dataset_name": "VinDr-Mammo", "description": "Mammography."},
    )

    assert json.loads(blueprint_path.read_text(encoding="utf-8"))["dataset"] == {
        "dataset_id": "vindr-mammo",
        "dataset_name": "VinDr-Mammo",
        "description": "Mammography.",
    }
