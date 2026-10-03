import json
from pathlib import Path

import pytest

from medbenchagent.planning.bir import BIRStore
from medbenchagent.runtime import sessions


def test_new_experiment_initializes_bir_directory_and_manifest(tmp_path: Path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    bir = tmp_path / "pilot" / "BIR"
    assert Path(record["bir_path"]) == bir
    assert (bir / "registry" / "axes.json").is_file()
    manifest = json.loads((bir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bir"
    assert manifest["revision"] == 0
    assert record["bir_manifest_digest"].startswith("sha256:")


def test_bir_revision_conflict_and_digest_validation(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    first = store.update("benchmark_info.json", {"schema_version": "benchmark-info", "value": 1},
                         run_id="r1", actor="test", expected_revision=0)
    assert first.revision == 1
    second = store.update("benchmark_info.json", {"schema_version": "benchmark-info", "value": 2},
                          run_id="r2", actor="test", expected_revision=0)
    assert second.revision == 2
    assert store.validate().valid
    (tmp_path / "pilot" / "BIR" / "benchmark_info.json").write_text("{}", encoding="utf-8")
    assert not store.validate().valid


def test_bir_rejects_path_escape(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    with pytest.raises(ValueError):
        store.load("../outside.json")


def test_bir_json_is_readable_and_failed_transaction_preserves_previous_file(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    store.update("allocation.json", {"schema_version": "allocation", "items": ["one"]},
                 run_id="r1", actor="test")
    path = store.root / "allocation.json"
    text = path.read_text(encoding="utf-8")
    assert "\n  \"items\"" in text
    before = json.loads(text)
    with pytest.raises(TypeError):
        store.update("allocation.json", {"schema_version": "allocation", "items": {1, 2}},
                     run_id="r2", actor="test")
    assert json.loads(path.read_text(encoding="utf-8")) == before

