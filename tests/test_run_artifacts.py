import json
import re
from pathlib import Path

from medbenchagent.runtime.run_artifacts import initialize_run


def test_initialize_run_prepares_uuid_directory_and_event_file(tmp_path: Path) -> None:
    run = initialize_run(tmp_path / "runs")

    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", run["run_id"])
    assert Path(run["construction_events_path"]).is_file()
    assert Path(run["working_blueprint_path"]).parent == tmp_path / "runs" / run["run_id"]
    event = json.loads(Path(run["construction_events_path"]).read_text(encoding="utf-8"))
    assert event["event_type"] == "run_started"
    assert event["component"] == "orchestrator"
    assert "agent" not in event
