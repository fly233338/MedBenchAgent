import json
from pathlib import Path

import duckdb
import pytest
from PIL import Image

from medbenchagent.data.population import verify_population
from medbenchagent.data.documentation import freeze_document, read_document
from medbenchagent.data.dataset_catalog import load_prepared
from medbenchagent.planning.scope import conditions_for
from medbenchagent.runtime.dependencies import publish


@pytest.mark.parametrize("actual", [
    [("c", "c.png"), ("d", "d.png")],
    [("a", "b.png"), ("b", "a.png")],
    [("a", "a.png"), ("a", "a.png")],
])
def test_equal_counts_do_not_hide_identity_changes(tmp_path, actual):
    expected = tmp_path / "population.jsonl"
    expected.write_text("".join(json.dumps({"source_id": "source", "image_id": key, "image_path": path}) + "\n"
                                 for key, path in [("a", "a.png"), ("b", "b.png")]), encoding="utf-8")
    emitted = tmp_path / "annotation.jsonl"
    emitted.write_text("".join(json.dumps({"record": {"image": {"image_id": key, "image_path": path}}}) + "\n"
                                for key, path in actual), encoding="utf-8")
    with pytest.raises(ValueError, match="population"):
        verify_population(tmp_path, expected, emitted, "source")


def test_cache_mutation_blocks_admission_and_reuse(tmp_path, monkeypatch):
    from test_dataset_cards import prepared_dataset
    from medbenchagent.data.dataset import prepare_annotation_dataset
    entry = prepared_dataset(tmp_path, "a", monkeypatch)
    _, access = load_prepared(entry)
    with duckdb.connect(access["duckdb_path"]) as db:
        db.execute("UPDATE images SET record='{}'")
    with pytest.raises(ValueError, match="DuckDB cache changed"):
        load_prepared(entry)
    with pytest.raises(ValueError, match="DuckDB cache changed"):
        prepare_annotation_dataset(access["annotation_path"], Path(access["annotation_path"]).parent / "cache")


def test_replaced_image_blocks_finalization_without_quarantining(tmp_path, monkeypatch):
    from test_run_export import export_context, call
    from medbenchagent.agents.finalization import inspect_dataset
    c = export_context(tmp_path, monkeypatch)
    entry = c["dataset_assets"]["a"]["entry"]
    manifest = Path(entry["image_manifest_path"])
    image = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    path = manifest.parent / image["png_path"]
    replacement = Image.new("L", (image["width"], image["height"]), 17)
    replacement.putpixel((0, 0), 199)
    replacement.save(path)
    with pytest.raises(ValueError, match="image integrity failure"):
        call(c, inspect_dataset, dataset_id="a")
    assert not (Path(c["run_directory"]) / "delivery/complete.json").exists()


def test_planning_document_is_independent_of_raw_directory(tmp_path):
    from medbenchagent.agents.task_planning import read_dataset_document, list_dataset_documents
    raw, bundle = tmp_path / "raw", tmp_path / "prepared"
    raw.mkdir()
    document = raw / "codebook.md"
    document.write_text("A means benign", encoding="utf-8")
    freeze_document(bundle, "source", raw, "codebook.md")
    document.write_text("A means malignant", encoding="utf-8")
    document.unlink()
    raw.rmdir()
    c = {"run_directory": str(tmp_path / "run"), "operation_directory": str(tmp_path / "op"),
         "dataset_scope": "a", "annotation_path": str(bundle / "annotation.jsonl"), "agent_role": "TaskPlanningAgent"}
    publish(c, "Dataset", {}, ())
    assert json.loads(list_dataset_documents(c)["summary"])["total"] == 1
    assert "A means benign" in read_dataset_document(c, path="source/codebook.md")["summary"]
    assert "malignant" not in read_dataset_document(c, path="source/codebook.md")["summary"]


def test_text_pdf_pages_are_frozen_and_blank_pdf_is_explicit(tmp_path):
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
    raw = tmp_path / "raw"
    raw.mkdir()
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 10 200 Td (A means benign) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(raw / "codes.pdf")
    freeze_document(tmp_path / "bundle", "s", raw, "codes.pdf")
    assert "A means benign" in read_document(tmp_path / "bundle", "s", "codes.pdf", page=1)["text"]
    blank = PdfWriter()
    blank.add_blank_page(width=300, height=300)
    blank.write(raw / "scan.pdf")
    with pytest.raises(ValueError, match="OCR is not supported"):
        freeze_document(tmp_path / "bundle", "s", raw, "scan.pdf")


def test_modality_union_cannot_be_split_into_intersection():
    requirement = [{"id": "m", "kind": "modality", "description": "CT or MRI", "quote": "CT or MRI",
                    "modalities": ["CT", "MRI"], "operator": "in"}]
    condition = {"requirement_id": "m", "field": "image.modality", "operator": "in", "values": ["CT", "MRI"]}
    resolution = {"requirement_id": "m", "status": "implemented", "reason": "Annotated modality", "conditions": [condition],
                  "modality_bindings": [{"modality": m, "values": [m]} for m in ("CT", "MRI")]}
    assert conditions_for({"scope_resolutions": [resolution]}, requirement) == [condition]
    resolution["conditions"] = [{**condition, "values": [m]} for m in ("CT", "MRI")]
    with pytest.raises(ValueError, match="exactly one"):
        conditions_for({"scope_resolutions": [resolution]}, requirement)
    resolution.update(status="unsupported", conditions=[], modality_bindings=[])
    with pytest.raises(ValueError, match="inspection references"):
        conditions_for({"scope_resolutions": [resolution]}, requirement)


def test_pairing_nested_fields_matches_resolver_and_rejects_other_array(tmp_path):
    from test_data_layer import record
    from test_generation import _task
    from medbenchagent.data.dataset import prepare_annotation_dataset
    from medbenchagent.data.query import DatasetQuery
    from medbenchagent.planning.protocol import resolve_answer
    row = record()
    row["image"]["annotation"] = {"findings": [{"details": {"name": "mass"}, "bbox": [0, 0, 2, 2]}],
                                  "other": [{"name": "wrong lesion"}]}
    path = tmp_path / "annotation.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _, access = prepare_annotation_dataset(path, tmp_path / "cache")
    name, bbox = "image.annotation.findings[*].details.name", "image.annotation.findings[*].bbox[*]"
    query = DatasetQuery(access)
    assert query.pair(name, bbox)["same_entity_pairs"] == 1
    protocol = _task("bbox", [name, bbox], target_field=name)["protocol"]
    assert resolve_answer(row, protocol)["target"] == "mass"
    wrong = "image.annotation.other[*].name"
    assert query.pair(wrong, bbox)["same_entity_pairs"] == 0
    protocol["data_binding"]["target_field"] = wrong
    assert resolve_answer(row, protocol) is None


def test_join_inspection_and_execution_share_composite_key_rules(tmp_path, monkeypatch):
    from medbenchagent.data.preparation_ops import inspect_join
    from medbenchagent.data.operators import inspect_table
    monkeypatch.setenv("MEDBENCH_DATA_SCRATCH", str(tmp_path / "scratch"))
    (tmp_path / "left.csv").write_text("patient;study;label\np;s1;A\np;s2;B\n", encoding="utf-8")
    (tmp_path / "right.csv").write_text("patient;study;value\np;s1;1\np;s2;2\n", encoding="utf-8")
    result = inspect_join(tmp_path, "left.csv", "right.csv", ["patient", "study"], ["patient", "study"], "one_to_one",
                          left_delimiter=";", right_delimiter=";")
    assert result["violations"] == [] and result["matched_left"] == 2
    wrong = inspect_join(tmp_path, "left.csv", "right.csv", ["patient"], ["patient"], "one_to_one",
                         left_delimiter=";", right_delimiter=";")
    assert "Join cardinality violated" in wrong["violations"]
    assert inspect_table(tmp_path, "left.csv", delimiter=";")["rows"] == 2
