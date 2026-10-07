import json
from pathlib import Path

import pytest
from PIL import Image

from medbenchagent.data.models import AssetEntry, PresentationSpec
from medbenchagent.data.presentation import target_path, materialize_presentation, verify_image
from medbenchagent.preparation.assembly import validate_stage_outputs, assemble_candidate
from medbenchagent.preparation.contracts import Stage, ArtifactRef, digest_file
from medbenchagent.preparation.validation import create_accepted_handoff, validate_understand_candidate, validate_prepare_candidate
from medbenchagent.preparation.publisher import DatasetPublisher
from test_preparation_validation import _understand_candidate


@pytest.mark.parametrize("format,suffix", [("JPEG", "jpg"), ("PNG", "png")])
def test_text_image_package_publish_preserves_native_bytes(tmp_path, format, suffix):
    source = tmp_path / "raw"
    source.mkdir()
    image = source / f"source.{suffix}"
    Image.new("RGB", (8, 6), "red").save(image, format=format)
    understand = _understand_candidate(tmp_path)
    (understand / "image_conversion_plan.json").write_text('{"policies":{"native":{}},"notes":[]}', encoding="utf-8")
    outputs = {p.name: str(p) for p in understand.iterdir()}
    state = {"dataset_id": "demo", "source_snapshot_digest": "snapshot", "source_roots": {"source": str(source)},
             "preparation_root": str(tmp_path / "runtime"), "bundle_root": str(tmp_path / "bundle")}
    result = validate_understand_candidate(outputs, known_source_ids=["source"])
    result.receipt_ref = str(tmp_path / "validation.json")
    handoff = create_accepted_handoff(state, outputs, result)
    state["accepted_handoff"] = handoff.model_dump(mode="json")
    text = tmp_path / "text"
    text.mkdir()
    case = {"case_id": "case", "evidence": [{"id": "image", "type": "asset", "name": "image", "value": "asset"}]}
    (text / "cases.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    (text / "provenance.jsonl").write_text("".join(json.dumps({"source_id": "source", "source_unit_id": f"unit-{i}",
        "case_ids": ["case"], "record_refs": [f"row-{i}"], "status": "retained", "reason": "mapped"}) + "\n" for i in (1, 2)), encoding="utf-8")
    target = target_path("source", "asset", None, {}, suffix)
    (text / "image_requests.jsonl").write_text(json.dumps({"case_id": "case", "asset_id": "asset", "source_id": "source",
        "source_path": image.name, "source_format": suffix, "policy_id": "native", "order": 0, "target_path": target}) + "\n", encoding="utf-8")
    (text / "submission.json").write_text('{"requested_outcome":"submitted"}', encoding="utf-8")
    assert validate_stage_outputs(text, Stage.PREPARE_TEXT, state).status == "accepted"
    state["text_root"] = str(text)
    state["text_artifacts"] = {p.name: ArtifactRef(name=p.name, path=str(p), digest=digest_file(p), producer="text", stage="prepare_text").model_dump(mode="json") for p in text.iterdir()}
    attempt = tmp_path / "image"
    images = attempt / "outputs"
    images.mkdir(parents=True)
    asset = AssetEntry(asset_id="asset", source_id="source", kind="image", format=suffix, source_path=image.name,
                       file_digest=digest_file(image), metadata={"input_visibility": "model_visible"})
    info = materialize_presentation(asset, PresentationSpec(), images / target, source_root=source)
    (images / "asset_manifest.jsonl").write_text(asset.model_dump_json() + "\n", encoding="utf-8")
    record = {"schema_version": "presentation-record", "presentation_id": "presentation", "case_id": "case", "asset_id": "asset",
              "path": target, "input_digest": asset.file_digest, "output_digest": info["output_digest"], "mode": "native", "order": 0}
    (images / "presentation_manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (images / "mask_features.jsonl").write_text("", encoding="utf-8")
    (images / "submission.json").write_text('{"requested_outcome":"submitted"}', encoding="utf-8")
    state["attempt_workspace"] = str(attempt)
    assert validate_stage_outputs(images, Stage.PREPARE_IMAGE, state).status == "accepted"
    package = assemble_candidate(state)
    validation = validate_prepare_candidate(package, state=state, population_path=handoff.population.path)
    assert validation.status == "accepted"
    published = DatasetPublisher(state["bundle_root"]).publish(package, state, validation)
    assert (Path(published["revision_path"]) / target).read_bytes() == image.read_bytes()
    assert not (Path(published["revision_path"]) / "submission.json").exists()
    (images / target).write_bytes(b"bad image")
    assert validate_stage_outputs(images, Stage.PREPARE_IMAGE, state).status == "rejected"


def test_image_encoding_cannot_be_faked_by_extension(tmp_path):
    path = tmp_path / "fake.png"
    Image.new("RGB", (2, 2)).save(path, format="JPEG")
    with pytest.raises(ValueError, match="encoding"):
        verify_image(path)


def test_shared_target_identity_keeps_views_distinct():
    assert target_path("s", "a", "one", {}, "nifti") != target_path("s", "a", "two", {}, "nifti")
    assert target_path("s", "a", None, {}, "jpeg").endswith(".jpeg")
