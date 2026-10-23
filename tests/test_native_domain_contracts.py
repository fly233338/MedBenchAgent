"""Dataset content identity and closed-VQA execution contract checks."""

import json

import pytest

from medbenchagent.compilation.artifacts import resolve_artifact
from dataset_fixtures import create_synthetic_inputs
from medbenchagent.data.dataset import FileDatasetReader, snapshot_dataset


def test_frozen_dataset_members_and_portable_snapshot(tmp_path):
    ref = create_synthetic_inputs(tmp_path / "original", "data")[0]
    source = resolve_artifact(tmp_path / "original", ref)
    copied = snapshot_dataset(source, tmp_path / "copy")
    reader = FileDatasetReader(copied)
    mtimes = {path: path.stat().st_mtime_ns for path in copied.parent.rglob("*") if path.is_file()}
    assert snapshot_dataset(source, copied.parent) == copied
    assert all(path.stat().st_mtime_ns == time for path, time in mtimes.items())
    assert reader.get_case("case-a").case_id == "case-a"
    for relative in (
        "cases.jsonl",
        "mapping.json",
        "sources/raw/image-a.png",
        "sources/raw/source.json",
    ):
        path = copied.parent / relative
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="content changed"):
            reader.get_case("case-a")
        path.write_bytes(original)
    (source.parent / "source.json").write_text("changed", encoding="utf-8")
    assert reader.get_case("case-a").case_id == "case-a"
    raw = json.loads(copied.read_text())
    raw["members"].pop()
    copied.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="cover exactly"):
        FileDatasetReader(copied)
