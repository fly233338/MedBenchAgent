"""Offline data-engine fixtures; no model calls, Docker runs, or real datasets."""

import json
from pathlib import Path

import pytest
from PIL import Image

from medbenchagent.data.bundle import normalize_source, publish_bundle
from medbenchagent.data.contracts import PreparationPlan, Source, HarmonizationPlan
from medbenchagent.data.operators import execute_plan, inspect_table
from medbenchagent.data.query import DatasetQuery, iter_records
from medbenchagent.data.sources import inventory_source, track_files, verify_source
from medbenchagent.runtime.dependencies import publish, require


def record(identifier="01", label="benign", path="image.png"):
    return {
        **{e: {e + "_id": None, "annotation": {}} for e in ("patient", "case", "study")},
        "image": {
            "image_id": identifier,
            "image_path": path,
            "modality": "US",
            "anatomy": "breast",
            "annotation": {"label": label},
        },
    }


def source_fixture(root, source_id="a", rows=None, pixel=123):
    root.mkdir(parents=True)
    image = Image.new("L", (8, 8))
    image.putpixel((2, 2), pixel)
    image.save(root / "image.png")
    (root / "annotation.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (rows or [record()])), encoding="utf-8"
    )
    return Source(id=source_id, root=str(root), kind="annotation")


def test_source_versions_do_not_overwrite_each_other(tmp_path):
    context = {"run_directory": str(tmp_path)}
    for source in ("a", "b"):
        context["source_scope"] = source
        publish(context, "Source", {"path": source}, ())
        publish(context, "NormalizedSource", {"path": source + "/normalized"}, ("Source",))
    publish(
        context, "Harmonization", {}, ("source/a/NormalizedSource", "source/b/NormalizedSource")
    )
    require(context, "Harmonization")
    context["source_scope"] = "a"
    publish(context, "Source", {"path": "changed"}, ())
    with pytest.raises(ValueError, match="stale"):
        require(context, "Harmonization")
    require(context, "source/b/NormalizedSource")


def test_source_manifest_detects_content_and_population_change(tmp_path):
    source = source_fixture(tmp_path / "raw")
    manifest = inventory_source(source, tmp_path / "manifest")
    track_files(manifest, ["annotation.jsonl"])
    verify_source(manifest)
    (Path(source.root) / "new.csv").write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="population changed"):
        verify_source(manifest)


def plan(steps, count=1):
    return PreparationPlan.model_validate(
        {
            "mapping": {
                "authoritative_population": "table rows",
                "expected_images": count,
                "entity_meanings": {"image": "one row per image"},
                "evidence": ["README"],
            },
            "steps": steps,
        }
    )


def test_operator_mapping_preserves_leading_zero_and_boolean(tmp_path):
    (tmp_path / "labels.csv").write_text("id,path,label\n001,image.png,yes\n", encoding="utf-8")
    source = Source(id="a", root=str(tmp_path))
    p = plan(
        [
            {"op": "read", "path": "labels.csv", "output": "raw"},
            {
                "op": "value_mapping",
                "input": "raw",
                "output": "mapped",
                "field": "label",
                "mapping": [{"value": "yes", "target": True}],
            },
            {
                "op": "emit",
                "input": "mapped",
                "fields": {
                    "image.image_id": "id",
                    "image.image_path": "path",
                    "image.annotation.positive": "label",
                },
            },
        ]
    )
    execute_plan(tmp_path, source, p, tmp_path / "out")
    row = json.loads((tmp_path / "out/annotation.jsonl").read_text())["record"]
    assert row["image"]["image_id"] == "001"
    assert row["image"]["annotation"]["positive"] is True
    assert inspect_table(tmp_path, "labels.csv")["columns"]["id"]["type"] == "VARCHAR"


def test_join_cardinality_and_unmapped_values_block(tmp_path):
    (tmp_path / "a.csv").write_text("key,label\n1,A\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("key,value\n1,x\n1,y\n", encoding="utf-8")
    p = plan(
        [
            {"op": "read", "path": "a.csv", "output": "a"},
            {"op": "read", "path": "b.csv", "output": "b"},
            {
                "op": "join",
                "input": "a",
                "right": "b",
                "output": "joined",
                "left_keys": ["key"],
                "right_keys": ["key"],
                "cardinality": "many_to_one",
            },
            {"op": "emit", "input": "joined", "fields": {}},
        ]
    )
    with pytest.raises(ValueError, match="cardinality"):
        execute_plan(tmp_path, Source(id="a", root=str(tmp_path)), p, tmp_path / "out")


def test_bundle_quarantine_source_namespace_and_full_vocabulary(tmp_path):
    bundle = tmp_path / "bundle"
    sources = {}
    for name, pixel in (("a", 50), ("b", 200)):
        source = source_fixture(
            tmp_path / name, name, [record(label=name), record("bad", path="missing.png")], pixel
        )
        folder = bundle / "sources" / name
        normalize_source(source.root, Path(source.root) / "annotation.jsonl", source, folder)
        sources[name] = folder
    profile, access = publish_bundle(bundle, sources, HarmonizationPlan())
    assert profile["total_images"] == 2
    assert profile["summary"]["quality"]["quarantined"] == 2
    assert len({r["image"]["image_id"] for r in iter_records(access)}) == 2
    query = DatasetQuery(access)
    field = "image.annotation.sources.a.label"
    assert query.values(field)["values"] == [{"value": "a", "count": 1}]
    assert query.by_source(field)["sources"][1]["coverage_rate"] == 0
    for row in iter_records(access):
        assert (bundle / row["image"]["image_path"]).is_file()


def test_duplicate_conflict_cannot_silently_choose_a_label(tmp_path):
    source = source_fixture(
        tmp_path / "source", rows=[record("a", "benign"), record("b", "malignant", "copy.png")]
    )
    import shutil

    shutil.copyfile(Path(source.root) / "image.png", Path(source.root) / "copy.png")
    folder = tmp_path / "bundle/sources/a"
    normalize_source(source.root, Path(source.root) / "annotation.jsonl", source, folder)
    with pytest.raises(ValueError, match="No usable"):
        publish_bundle(tmp_path / "bundle", {"a": folder}, HarmonizationPlan())
    issues = (tmp_path / "bundle/quarantine.jsonl").read_text()
    assert issues.count("duplicate_conflict") == 2
    assert not (tmp_path / "bundle/ready.json").exists()


def test_array_pairing_and_capacity_share_generation_resolver(tmp_path):
    from medbenchagent.data.dataset import prepare_annotation_dataset
    from medbenchagent.planning.protocol import resolve_answer
    from test_generation import _task

    rows = [record(str(i), label=str(i)) for i in range(5)]
    for row in rows:
        row["image"]["image_path"] = row["image"]["image_id"] + ".png"
        row["image"]["annotation"]["findings"] = [{"name": "mass"}, {"bbox": [0, 0, 2, 2]}]
    path = tmp_path / "annotation.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _, access = prepare_annotation_dataset(path, tmp_path / "cache")
    query = DatasetQuery(access)
    first = query.values("image.annotation.label", limit=3)
    assert first["total"] == 5 and first["next_offset"] == 3
    assert query.values("image.annotation.label", offset=3, limit=3)["complete"]
    pair = query.pair("image.annotation.findings[*].name", "image.annotation.findings[*].bbox[*]")
    assert pair["same_image_count"] == 5 and pair["same_entity_pairs"] == 0
    protocol = _task("letter", ["image.annotation.status"])["protocol"]
    assert query.capacity(protocol)["eligible"] == sum(
        resolve_answer(r, protocol) is not None for r in rows
    )


def test_harmonization_preserves_array_instances_and_rejects_absent_values():
    from medbenchagent.data.bundle import align_record
    from medbenchagent.data.sources import attach_provenance

    row = record()
    row["image"]["annotation"] = {
        "findings": [{"name": "A", "bbox": [0, 0, 1, 1]}, {"name": "B", "bbox": [2, 2, 3, 3]}]
    }
    source = Source(id="a", root="unused")
    attach_provenance(row, source, "record:1", "plan")
    plan = HarmonizationPlan.model_validate(
        {
            "fields": [
                {
                    "source_id": "a",
                    "field": "image.annotation.findings[*].name",
                    "target": "image.annotation.findings[*].name",
                    "action": "merge",
                    "evidence": "documented codebook",
                    "mapping": [{"value": "A", "target": "mass"}, {"value": "B", "target": "cyst"}],
                }
            ]
        }
    )
    aligned = align_record(row, plan, composite=False, observed=set())
    assert aligned["image"]["annotation"]["findings"] == [
        {"name": "mass", "bbox": [0, 0, 1, 1]},
        {"name": "cyst", "bbox": [2, 2, 3, 3]},
    ]


def test_group_findings_retains_lineage_and_bbox_pairing(tmp_path):
    from medbenchagent.data.lineage import validate_lineage

    (tmp_path / "findings.csv").write_text(
        "id,path,name,x,y,w,h\n1,image.png,A,0,0,2,3\n1,image.png,B,4,4,1,1\n", encoding="utf-8"
    )
    p = plan(
        [
            {"op": "read", "path": "findings.csv", "output": "raw"},
            {
                "op": "bbox",
                "input": "raw",
                "output": "boxes",
                "fields": ["x", "y", "w", "h"],
                "target": "bbox",
                "convention": "xywh",
            },
            {
                "op": "group_findings",
                "input": "boxes",
                "output": "grouped",
                "keys": ["id", "path"],
                "fields": {"name": "name", "bbox": "bbox"},
                "target": "findings",
            },
            {
                "op": "emit",
                "input": "grouped",
                "fields": {
                    "image.image_id": "id",
                    "image.image_path": "path",
                    "image.annotation.findings": "findings",
                },
            },
        ]
    )
    output = tmp_path / "out"
    execute_plan(tmp_path, Source(id="a", root=str(tmp_path)), p, output)
    assert validate_lineage(output / "annotation.jsonl", output / "provenance.jsonl", tmp_path) == 1
    row = json.loads((output / "annotation.jsonl").read_text())["record"]
    assert row["image"]["annotation"]["findings"] == [
        {"name": "A", "bbox": [0, 0, 2, 3]},
        {"name": "B", "bbox": [4, 4, 5, 5]},
    ]


def test_header_only_dicom_inspection_does_not_request_pixels(monkeypatch):
    import sys
    from types import SimpleNamespace
    from medbenchagent.data.readers import DicomReader

    calls = []

    def read(path, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            PatientID="id", Rows=12, Columns=8, file_meta=SimpleNamespace(TransferSyntaxUID="1.2.3")
        )

    monkeypatch.setitem(sys.modules, "pydicom", SimpleNamespace(dcmread=read))
    result = DicomReader().read_metadata("unused.dcm")
    assert result["Rows"] == 12
    assert calls[0]["stop_before_pixels"] is True
    assert "PatientID" in calls[0]["specific_tags"]


def test_unmapped_scalar_does_not_become_missing(tmp_path):
    (tmp_path / "labels.csv").write_text("value\nyes\nunknown\n", encoding="utf-8")
    p = plan(
        [
            {"op": "read", "path": "labels.csv", "output": "a"},
            {
                "op": "value_mapping",
                "input": "a",
                "output": "b",
                "field": "value",
                "mapping": [{"value": "yes", "target": True}],
            },
            {"op": "emit", "input": "b", "fields": {}},
        ]
    )
    with pytest.raises(ValueError, match="Unmapped values"):
        execute_plan(tmp_path, Source(id="a", root=str(tmp_path)), p, tmp_path / "out")


def test_all_sources_required_and_atomic_ready_marker(tmp_path, monkeypatch):
    from dataclasses import replace
    from medbenchagent import settings
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths, prepared_datasets_root=tmp_path / "prepared"), data_processing={**cfg.data_processing, "backend": "local"}))
    from medbenchagent.agents.data_pipeline import initialize, save_harmonization

    context = initialize(
        {"run_directory": str(tmp_path), "operation_directory": str(tmp_path / "op")},
        [{"id": "a", "root": str(tmp_path / "a")}, {"id": "b", "root": str(tmp_path / "b")}],
        "both",
    )
    context["normalized_sources"] = {"a": "unused"}
    with pytest.raises(ValueError, match="Every selected source"):
        save_harmonization(context, plan={})
    source = source_fixture(tmp_path / "source")
    folder = tmp_path / "bundle/sources/a"
    normalize_source(source.root, Path(source.root) / "annotation.jsonl", source, folder)

    def changed():
        raise ValueError("source changed during publication")

    with pytest.raises(ValueError, match="source changed"):
        publish_bundle(
            tmp_path / "bundle", {"a": folder}, HarmonizationPlan(), verify_inputs=changed
        )
    assert not (tmp_path / "bundle/ready.json").exists()


def test_raw_and_standard_sources_through_agent_tools(tmp_path, monkeypatch):
    """Execute actual operators with a local container double, never a model or Docker."""
    from medbenchagent.agents import data_pipeline as agent
    from medbenchagent.agents import data_processing
    from medbenchagent.data.image_inventory import inspect_population
    from dataclasses import replace
    from medbenchagent import settings
    cfg = settings.get_settings()
    monkeypatch.setattr(settings, "_SETTINGS", replace(cfg, paths=replace(cfg.paths, prepared_datasets_root=tmp_path / "prepared", dataset_catalog=tmp_path / "catalog.json"), data_processing={**cfg.data_processing, "backend": "local"}))
    monkeypatch.setattr("medbenchagent.data.dataset_card.complete_json", lambda *a: {"description": "Test dataset"})

    raw = source_fixture(tmp_path / "raw", pixel=60)
    (Path(raw.root) / "labels.csv").write_text(
        "id,path,label\n01,image.png,benign\n", encoding="utf-8"
    )
    standard = source_fixture(tmp_path / "standard", "b", pixel=190)
    context = {
        "run_directory": str(tmp_path / "run"),
        "operation_directory": str(tmp_path / "run/op"),
        "user_requirement": "Test offline data preparation",
    }
    publish(context, "DataConfig", {}, ())
    context = agent.initialize(
        context, [{"id": "a", "root": raw.root, "kind": "raw"}, standard.model_dump()], "combined"
    )
    calls = []

    def local_run(c, output, *, request, annotation=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        operation = request["operation"]
        calls.append(operation)
        if operation == "table":
            return inspect_table(c["source_root"], request["path"])
        if operation == "execute_plan":
            return execute_plan(
                c["source_root"],
                Source.model_validate(request["source"]),
                PreparationPlan.model_validate(request["plan"]),
                output,
            )
        if operation == "inspect_population":
            return inspect_population(
                c["source_root"], annotation, output, wrapped=request["wrapped"]
            )
        if operation == "normalize":
            return normalize_source(
                c["source_root"],
                annotation,
                Source.model_validate(request["source"]),
                output,
                policies=request["policies"],
                wrapped=request["wrapped"],
                plan_ref=request["plan_ref"],
            )
        raise AssertionError(operation)

    monkeypatch.setattr(data_processing, "_run", local_run)
    index = 0

    def call(function, **arguments):
        nonlocal index
        index += 1
        context["operation_directory"] = str(tmp_path / "run" / f"op{index}")
        outcome = function(context, **arguments)
        context.update(outcome["updates"])
        return outcome

    call(agent.select_source, source_id="a")
    call(data_processing.inspect_table, path="labels.csv")
    preparation = plan(
        [
            {"op": "read", "path": "labels.csv", "output": "raw"},
            {
                "op": "emit",
                "input": "raw",
                "fields": {
                    "image.image_id": "id",
                    "image.image_path": "path",
                    "image.annotation.label": "label",
                },
                "constants": {"image.modality": "US", "image.anatomy": "breast"},
            },
        ]
    )
    call(agent.save_preparation_plan, plan=preparation.model_dump())
    call(agent.execute_preparation)
    call(agent.inspect_images)
    call(data_processing.save_image_plan, overrides={})
    call(agent.normalize_images)
    call(agent.select_source, source_id="b")
    call(agent.inspect_images)
    call(data_processing.save_image_plan, overrides={})
    call(agent.normalize_images)
    call(
        agent.save_harmonization,
        plan={
            "fields": [
                {
                    "source_id": source,
                    "field": "image.annotation.label",
                    "action": "merge",
                    "target": "image.annotation.label",
                    "evidence": "same codebook",
                }
                for source in ("a", "b")
            ]
        },
    )
    assert call(agent.finish)["status"] == "complete"
    assert call(agent.finish)["status"] == "complete"  # publication replay is idempotent
    assert DatasetQuery(context["dataset_access_manifest"]).values("image.annotation.label")[
        "values"
    ] == [{"value": "benign", "count": 2}]
    assert calls.count("normalize") == 4 and calls.count("execute_plan") == 2


def test_uint16_assets_are_not_misclassified_as_uniform(tmp_path):
    import numpy as np
    from medbenchagent.data.image_assets import validate_image_asset

    pixels = np.arange(1024, 1088, dtype=np.uint16).reshape(8, 8)
    Image.fromarray(pixels).save(tmp_path / "image.png")
    assert validate_image_asset({"png_path": "image.png", "width": 8, "height": 8}, tmp_path) == []


def test_standard_import_quarantines_missing_image_and_keeps_valid_record(tmp_path):
    from medbenchagent.runtime.dependencies import snapshot_dataset

    source = source_fixture(
        tmp_path / "source", rows=[record(), record("missing", path="missing.png")]
    )
    root = Path(source.root)
    (root / "image_manifest.jsonl").write_text(
        "".join(
            json.dumps({"image_id": identifier, "png_path": path, "width": 8, "height": 8}) + "\n"
            for identifier, path in [("01", "image.png"), ("missing", "missing.png")]
        ),
        encoding="utf-8",
    )
    published = snapshot_dataset(root / "annotation.jsonl", tmp_path / "published", dataset_id="a")
    quality = json.loads((published.parent / "quality_report.json").read_text())
    assert quality["accepted"] == 1 and quality["quarantined"] == 1


def test_output_write_failure_blocks_instead_of_quarantining(tmp_path, monkeypatch):
    from medbenchagent.data.bundle import OutputWriteError

    source = source_fixture(tmp_path / "source")

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("medbenchagent.data.bundle.shutil.copyfile", fail)
    with pytest.raises(OutputWriteError, match="disk full"):
        normalize_source(
            source.root, Path(source.root) / "annotation.jsonl", source, tmp_path / "out"
        )


def test_mask_to_bbox_preserves_instances_and_empty_masks(tmp_path):
    import numpy as np

    pixels = np.zeros((8, 8), dtype=np.uint8)
    pixels[1:3, 2:4] = 1
    pixels[5:7, 5:8] = 2
    Image.fromarray(pixels).save(tmp_path / "mask.png")
    (tmp_path / "masks.csv").write_text("id,path,width,height\n1,mask.png,8,8\n", encoding="utf-8")
    steps = [
        {"op": "read", "path": "masks.csv", "output": "raw"},
        {
            "op": "mask_to_bbox",
            "input": "raw",
            "output": "boxes",
            "field": "path",
            "target": "boxes",
            "foreground": [1, 2],
            "instances": "by_value",
            "width_field": "width",
            "height_field": "height",
        },
        {
            "op": "emit",
            "input": "boxes",
            "fields": {
                "image.image_id": "id",
                "image.image_path": "path",
                "image.annotation.boxes": "boxes",
            },
        },
    ]
    execute_plan(tmp_path, Source(id="a", root=str(tmp_path)), plan(steps), tmp_path / "out")
    row = json.loads((tmp_path / "out/annotation.jsonl").read_text())["record"]
    assert row["image"]["annotation"]["boxes"] == [[2, 1, 4, 3], [5, 5, 8, 7]]
