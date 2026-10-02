import json

import pytest

from medbenchagent.runtime import dependencies


def _context(tmp_path):
    run = tmp_path / "run"
    control = run / "control"
    control.mkdir(parents=True)
    frozen = control / "effective_config.json"
    frozen.write_text(json.dumps({"execution_runtime": {"backend": "local"}}), encoding="utf-8")
    return {
        "run_directory": str(run),
        "effective_config_path": str(frozen),
        "execution_runtime": {"backend": "local"},
        "task_catalog_snapshot_path": str(control / "catalog.json"),
        "task_catalog_snapshot_digest": "digest",
        "construction_events_path": str(control / "events.jsonl"),
        "run_id": "run-1",
    }


def test_resume_allows_source_changes_but_records_diagnostic(tmp_path, monkeypatch):
    context = _context(tmp_path)
    start = {"commit": "old", "source_diff": "", "untracked_sources": {}}
    current = {"commit": "new", "source_diff": "changed", "untracked_sources": {}}
    (tmp_path / "run" / "control" / "contract.json").write_text(
        json.dumps({"contract": dependencies.CONTRACT_VERSION, "code_at_start": start}),
        encoding="utf-8",
    )
    monkeypatch.setattr(dependencies, "code_identity", lambda: current)
    monkeypatch.setattr("medbenchagent.runtime.run_artifacts.verify_catalog_snapshot", lambda *_args: None)
    dependencies.verify_resume(context)
    events = [json.loads(line) for line in (tmp_path / "run" / "control" / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "resume_compatibility"
    assert events[-1]["code_changed"] is True


def test_resume_still_rejects_contract_change(tmp_path):
    context = _context(tmp_path)
    (tmp_path / "run" / "control" / "contract.json").write_text(
        json.dumps({"contract": "old-contract", "code_at_start": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contract changed"):
        dependencies.verify_resume(context)
