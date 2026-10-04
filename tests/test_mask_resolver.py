import hashlib
import pytest
from PIL import Image

from medbenchagent.audit.task_items import audit_task_items
from medbenchagent.data.asset_manifest import write_asset_manifest
from medbenchagent.data.models import AssetEntry
from medbenchagent.generation.items import generate_task_items
from medbenchagent.planning.resolver import ResolverContext, ResolverSpec, resolve
from medbenchagent.planning.task_space import CapabilityCategory, OutputSpec, TaskParadigm


def _mask_context(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    mask_path = bundle / "mask.png"
    image = Image.new("RGBA", (10, 8), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(2, 6):
        for x in range(3, 8):
            pixels[x, y] = (255, 255, 255, 255)
    image.save(mask_path)
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    manifest = bundle / "asset_manifest.jsonl"
    write_asset_manifest(manifest, [AssetEntry(
        asset_id="mask-1", kind="image", format="png", source_path="mask.png",
        file_digest=digest, width=10, height=8,
    )])
    return ResolverContext.from_access({
        "asset_manifest_path": str(manifest),
        "cases_path": str(bundle / "cases.jsonl"),
    })


def _case():
    return {"case_id": "c1", "evidence": [
        {"id": "image", "type": "asset", "name": "ultrasound_bmode_image", "value": "image-1", "ref": []},
        {"id": "mask", "type": "asset", "name": "tumor_mask", "value": "mask-1", "ref": ["image"]},
    ]}


def _spec():
    return ResolverSpec(
        inputs=[{"role": "image", "name": "ultrasound_bmode_image", "evidence_type": "asset"}],
        steps=[
            {"op": "read", "selector": {"role": "mask", "name": "tumor_mask", "evidence_type": "asset", "relation": "image"}},
            {"op": "mask_to_bbox", "input": "step_0", "channel": "alpha", "foreground": [255]},
        ],
        output_spec=OutputSpec.BBOX,
        result_step="step_1",
    )


def test_mask_to_bbox_is_deterministic_and_normalized(tmp_path):
    result = resolve(_case(), _spec(), resolver_context=_mask_context(tmp_path))

    assert result.reference == [0.3, 0.25, 0.8, 0.75]
    assert result.input_evidence_ids == ["image"]
    assert result.reference_evidence_ids == ["mask"]


def test_mask_to_bbox_requires_asset_context(tmp_path):
    result = resolve(_case(), _spec())
    assert result.reason == "missing_asset_context"


def test_mask_to_bbox_reports_empty_and_digest_errors(tmp_path):
    context = _mask_context(tmp_path)
    empty_path = tmp_path / "bundle" / "empty.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(empty_path)
    context.assets["mask-1"]["source_path"] = "empty.png"
    context.assets["mask-1"]["file_digest"] = hashlib.sha256(empty_path.read_bytes()).hexdigest()
    context.assets["mask-1"]["width"] = 4
    context.assets["mask-1"]["height"] = 4
    context.cache.clear()
    result = resolve(_case(), _spec(), resolver_context=context)
    assert result.reason == "mask_empty"

    context.assets["mask-1"]["file_digest"] = "wrong"
    context.cache.clear()
    result = resolve(_case(), _spec(), resolver_context=context)
    assert result.reason == "asset_digest_mismatch"


def test_bbox_resolver_rejects_asset_locate_and_invalid_mask_input():
    with pytest.raises(ValueError, match="locate requires a region or index"):
        ResolverSpec(
            inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
            steps=[
                {"op": "read", "selector": {"role": "mask", "name": "mask", "evidence_type": "asset"}},
                {"op": "locate", "input": "step_0"},
            ],
            output_spec=OutputSpec.BBOX,
            result_step="step_1",
        )


def test_generation_and_audit_keep_mask_out_of_model_inputs(tmp_path):
    leaf = {
        "leaf_id": "bbox-leaf", "name": "tumor location", "node_id": "scenario-1",
        "clinical_target_id": "finding", "question_templates": ["Locate the tumor."],
        "input": [{"role": "image", "name": "ultrasound_bmode_image", "evidence_type": "asset"}],
        "target": "tumor location", "reference_rule": "Use the registered mask bbox.",
        "resolver": _spec().model_dump(mode="json"), "output_spec": "bbox",
        "task_paradigm": TaskParadigm.VISUAL_GROUNDING.value,
        "capability_category": CapabilityCategory.LOCALIZATION.value,
    }
    context = _mask_context(tmp_path)
    items = generate_task_items([_case()], [leaf], "dataset", resolver_context=context)
    assert len(items) == 1
    assert items[0]["answer"] == [0.3, 0.25, 0.8, 0.75]
    assert items[0]["input_assets"] == ["image-1"]
    assert audit_task_items([_case()], items, [leaf], resolver_context=context)["accepted"] == 1
