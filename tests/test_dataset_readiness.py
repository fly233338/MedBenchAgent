import json
from dataclasses import replace
from pathlib import Path

import pytest

from medbenchagent import settings
from medbenchagent.data.readiness import dataset_status, list_datasets, reuse_prepared
from medbenchagent.data.dataset_catalog import load_dataset_catalog
from medbenchagent.runtime.dependencies import publish, require
from test_dataset_cards import prepared_dataset


def setup(tmp_path, monkeypatch):
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths, dataset_catalog=tmp_path / "catalog.json"),
        dataset_sources=({"dataset_id": "a", "name": "A", "description": "Raw data",
                          "sources": [{"id": "raw", "root": str(tmp_path / "missing"), "kind": "raw"}]},)))
    return {"run_directory": str(tmp_path / "run"), "operation_directory": str(tmp_path / "run/reuse")}


def test_observation_is_read_only_and_reuse_does_not_need_raw_source(tmp_path, monkeypatch):
    c = setup(tmp_path, monkeypatch)
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    status = dataset_status(c, "a")
    assert status["prepared"] and status["source_errors"] and status["next_action"] == "reuse"
    assert json.loads(list_datasets(c, limit=1)["summary"])["total"] == 1
    assert not (tmp_path / "run").exists()
    publish(c, "DataConfig", {}, ())
    c.update(reuse_prepared(c, "a")["updates"])
    require(c, "dataset/a/DatasetCard", "PreparedCandidates")
    frozen = c["prepared_entries"]["a"]["dataset_card_path"]
    assert frozen != entry["dataset_card_path"]
    Path(entry["dataset_card_path"]).unlink()
    assert reuse_prepared(c, "a")["updates"]["prepared_entries"]["a"]["dataset_card_path"] == frozen


def test_absent_catalog_is_empty_but_corrupt_catalog_fails(tmp_path, monkeypatch):
    c = setup(tmp_path, monkeypatch)
    assert load_dataset_catalog() == []
    assert dataset_status(c, "a")["next_action"] == "fix_source_configuration"
    (tmp_path / "catalog.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="datasets array"):
        list_datasets(c)


@pytest.mark.parametrize("disabled", [True, False])
def test_disabled_or_invalid_dataset_needs_repair(tmp_path, monkeypatch, disabled):
    c = setup(tmp_path, monkeypatch)
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    if disabled:
        entry["available"] = False
        (tmp_path / "catalog.json").write_text(json.dumps({"datasets": [entry]}), encoding="utf-8")
    else:
        Path(entry["profile_path"]).unlink()
    status = dataset_status(c, "a")
    assert not status["prepared"] and status["issues"]
    assert status["next_action"] == "repair_catalog_or_explicit_rebuild"
