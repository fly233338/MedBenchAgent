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
    result = compile_preparation_graph(worker=Worker()).invoke(state(tmp_path))
    root = Path(result["package_root"])
    if change == "missing_image":
        (root / "images/view.jpg").unlink()
    elif change == "changed_image":
        (root / "images/view.jpg").write_bytes(b"corrupt")
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
