import json
from pathlib import Path
import runpy

from breast_fixtures import create_breast_source
from medbenchagent.data.dataset import FileDatasetReader


def test_multiview_case_preserves_unknown_mask_links_and_missing_labels(tmp_path):
    source = tmp_path / "raw"
    workbook = create_breast_source(source)
    prepare = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_breast_example.py"))["prepare"]
    manifest = prepare(workbook, source, tmp_path / "canonical")
    reader = FileDatasetReader(manifest)
    cases = reader.list_cases().items
    assert len(cases) == 4
    multi = {row.id: row for row in cases[0].evidence}
    assert multi["pathology"].ref == ["image-0", "image-1"]
    assert multi["mask_tumor_filename-0"].ref == []
    single = {row.id: row for row in cases[1].evidence}
    assert single["mask_tumor_filename-0"].ref == ["image-0"]
    assert next(row for row in cases[3].evidence if row.name == "pathology").value is None
    mapping = json.loads((manifest.parent / "mapping.json").read_text(encoding="utf-8"))
    assert mapping["unknown_relations"][0]["case_id"] == "case-001"
