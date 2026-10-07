import json
from pathlib import Path
from test_preparation_graph import mapping, state, Worker
from medbenchagent.preparation.contracts import Stage, ValidationResult
from medbenchagent.preparation.validation import validate_stage, validate_bundle, bundle_digest
from medbenchagent.preparation.graph import compile_preparation_graph
from medbenchagent.preparation.publisher import DatasetPublisher
import pytest


def test_understand_has_only_two_semantic_outputs_and_rejects_bad_source(tmp_path):
    context = state(tmp_path)
    root = tmp_path / "understand"
    root.mkdir()
    (root / "mapping.json").write_text(json.dumps(mapping()), encoding="utf-8")
    (root / "image_conversion_plan.json").write_text('{"policies":{"native":{}}}', encoding="utf-8")
    before = {p.name:p.read_bytes() for p in root.iterdir()}
    assert validate_stage(root, Stage.UNDERSTAND, context).status == "accepted"
    assert before == {p.name:p.read_bytes() for p in root.iterdir()}
    (root / "mapping.json").write_text(json.dumps(mapping(filename="missing.jpg")), encoding="utf-8")
    assert validate_stage(root, Stage.UNDERSTAND, context).status == "rejected"


@pytest.mark.parametrize("change", ["missing_image", "duplicate_case", "orphan_image", "changed_image"])
def test_final_cross_check_rejects_inconsistent_dataset(tmp_path, change):
    result = compile_preparation_graph(worker=Worker(convert=True)).invoke(state(tmp_path))
    root = Path(result["package_root"])
    if change == "missing_image":
        Path(json.loads((root / "image_manifest.jsonl").read_text())["path"]).unlink()
    elif change == "changed_image":
        Path(json.loads((root / "image_manifest.jsonl").read_text())["path"]).write_bytes(b"corrupt")
    elif change == "duplicate_case":
        p = root / "cases.jsonl"
        p.write_bytes(p.read_bytes() * 2)
    else:
        p = root / "image_manifest.jsonl"
        row = json.loads(p.read_text())
        row["path"] = "images/orphan.jpg"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    with pytest.raises((ValueError, OSError)):
        validate_bundle(root)


def test_publisher_only_commits_unchanged_staging_and_is_idempotent(tmp_path):
    context = state(tmp_path)
    result = compile_preparation_graph(worker=Worker()).invoke(context)
    root = Path(result["package_root"])
    receipt = ValidationResult(status="accepted", stage=Stage.PREPARE_IMAGE, candidate_digest=bundle_digest(root))
    publisher = DatasetPublisher(context["bundle_root"])
    first = publisher.publish(root, context, receipt)
    assert publisher.publish(root, context, receipt) == first
    (root / "cases.jsonl").write_text("changed", encoding="utf-8")
    current = (Path(context["bundle_root"]) / "current.json").read_bytes()
    with pytest.raises(ValueError, match="unchanged"):
        publisher.publish(root, context, receipt)
    assert (Path(context["bundle_root"]) / "current.json").read_bytes() == current

def test_accounting_rejects_silent_truncation(tmp_path):
    from medbenchagent.preparation.accounting import validate_accounting
    from medbenchagent.preparation.contracts import PreparationMapping
    context = state(tmp_path)
    source = Path(context["source_roots"]["s"]) / "records.csv"
    source.write_text("id,label\n" + "c,yes\n" * 10000, encoding="utf-8")
    report = tmp_path / "accounting.json"
    report.write_text(json.dumps({"records_seen":6000, "records_consumed":6000, "cases_emitted":6000,
        "excluded":{}, "skipped":{}}), encoding="utf-8")
    with pytest.raises(ValueError, match="population"):
        validate_accounting(report, PreparationMapping.model_validate(mapping()), context["source_roots"], 6000)


def test_image_expansion_uses_frozen_inventory_and_series_parents(tmp_path, monkeypatch):
    from medbenchagent.data.contracts import Source
    from medbenchagent.data.sources import inventory_source
    from medbenchagent.preparation.contracts import PreparationMapping, ImageConversionPlan
    from medbenchagent.preparation.images import mapped_images

    context = state(tmp_path)
    raw = Path(context["source_roots"]["s"])
    series = raw / "study/series"
    series.mkdir(parents=True)
    (series / "slice.dcm").write_bytes(b"frozen slice")
    manifest = inventory_source(Source(id="s", root=str(raw)), tmp_path / "series_inventory")
    context["source_manifests"]["s"] = str(manifest)
    # Added after freezing: broad rules must not discover this image.
    (raw / "new.jpg").write_bytes((raw / "raw.jpg").read_bytes())
    value = mapping(filename="*.jpg")
    value["image_overrides"] = {"s:raw.jpg": {**value["image_rules"][0], "rule_id": "override"}}
    value["image_rules"].append({"rule_id": "series", "source_id": "s", "pattern": "study/*",
                                "source_format": "dicom_series", "policy_id": "native", "action": "convert"})
    original_rglob = Path.rglob

    def guarded_rglob(path, pattern):
        if path == raw:
            raise AssertionError("Source root must not be rescanned")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", guarded_rglob)
    images = mapped_images(PreparationMapping.model_validate(value),
                           ImageConversionPlan(policies={"native": {}}), context)

    assert {row[2] for row in images.values()} == {"raw.jpg", "study/series"}
    assert images[str(raw / "raw.jpg").replace("\\", "/")][0].rule_id == "override"
    assert len(images) == 2
