import json
from pathlib import Path

import pytest

from medbenchagent.planning.bir import BIRConflictError, BIRStore
from medbenchagent.runtime import sessions


def test_new_experiment_initializes_bir_directory_and_manifest(tmp_path: Path):
    record = sessions.create_experiment("pilot", root=tmp_path)
    bir = tmp_path / "pilot" / "BIR"
    assert Path(record["bir_path"]) == bir
    assert (bir / "registry" / "axes.json").is_file()
    manifest = json.loads((bir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bir.v1"
    assert manifest["revision"] == 0
    assert record["bir_manifest_digest"].startswith("sha256:")


def test_bir_revision_conflict_and_digest_validation(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    first = store.update("benchmark_info.json", {"schema_version": "benchmark-info.v1", "value": 1},
                         run_id="r1", actor="test", expected_revision=0)
    assert first.revision == 1
    with pytest.raises(BIRConflictError):
        store.update("benchmark_info.json", {"schema_version": "benchmark-info.v1", "value": 2},
                     run_id="r2", actor="test", expected_revision=0)
    assert store.validate().valid
    (tmp_path / "pilot" / "BIR" / "benchmark_info.json").write_text("{}", encoding="utf-8")
    assert not store.validate().valid


def test_bir_rejects_path_escape(tmp_path: Path):
    sessions.create_experiment("pilot", root=tmp_path)
    store = BIRStore(tmp_path / "pilot" / "BIR")
    with pytest.raises(ValueError):
        store.load("../outside.json")
