import math

import numpy as np
import pytest

from medbenchagent.data.asset_adapters import AssetAdapter, LoadedAsset
from medbenchagent.data.manifests import SamplingManifest, SamplingUnit
from medbenchagent.data.models import AssetEntry, CaseRecord, MaskFeatureSet, validate_case
from medbenchagent.planning.resolver import ResolverContext, ResolverSpec, resolve
from medbenchagent.planning.task_space import LeafTask


def _mask(digest="mask-digest"):
    return MaskFeatureSet(
        source_digest=digest,
        algorithm="prepared-v1",
        channel="alpha",
        foreground=[255],
        shape=[8, 10],
        bbox=[0.3, 0.25, 0.8, 0.75],
        pixel_count=20,
        area_fraction=0.25,
        centroid=[0.55, 0.5],
        component_count=1,
        equivalent_diameter_pixels=2 * math.sqrt(20 / math.pi),
        planes=[],
    )


def _case():
    return {
        "case_id": "case-1",
        "evidence": [
            {"id": "image", "type": "asset", "name": "image", "value": "image-1", "ref": [], "view_id": "v0"},
            {"id": "mask", "type": "asset", "name": "mask", "value": "mask-1", "ref": ["image"], "mask": _mask().model_dump(mode="json")},
            {"id": "area", "type": "value", "name": "area", "value": 0.25, "ref": ["image"], "view_id": "v0"},
        ],
    }


def test_mask_feature_is_digest_bound_and_resolves_without_file_access():
    case = CaseRecord.model_validate(_case())
    assets = {
        "image-1": AssetEntry(asset_id="image-1", kind="image", format="png", source_path="image.png", file_digest="image", width=10, height=8),
        "mask-1": AssetEntry(asset_id="mask-1", kind="image", format="png", source_path="mask.png", file_digest="mask-digest", width=10, height=8),
    }
    validate_case(case, assets)
    spec = ResolverSpec(
        inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
        steps=[
            {"op": "read", "selector": {"role": "mask", "name": "mask", "evidence_type": "asset", "relation": "image"}},
            {"op": "mask_to_bbox", "input": "step_0", "channel": "alpha", "foreground": [255]},
        ],
        output_spec="bbox",
        result_step="step_1",
    )
    result = resolve(case.model_dump(mode="json"), spec)
    assert result.reference == [0.3, 0.25, 0.8, 0.75]
    assert result.reference_evidence_ids == ["mask"]

    bad = case.model_copy(update={"evidence": [case.evidence[0], case.evidence[1].model_copy(update={"mask": _mask("wrong")}), case.evidence[2]]})
    with pytest.raises(ValueError, match="source_digest"):
        validate_case(bad, assets)


def test_argmax_requires_frozen_order_and_rejects_ties():
    case = _case()
    spec = ResolverSpec(
        inputs=[{"role": "image", "name": "image", "evidence_type": "asset"}],
        steps=[
            {"op": "collect", "selector": {"role": "area", "name": "area", "evidence_type": "value"}},
            {"op": "argmax", "input": "step_0"},
        ],
        output_spec="index",
        result_step="step_1",
    )
    manifest = SamplingManifest(
        dataset_id="dataset",
        revision="1",
        units=[SamplingUnit(sampling_unit_id="unit-1", case_id="case-1", split="test", view_ids=["v0"], ordered_evidence_ids=["area", "area-2"], eligible=True, order=0)],
    )
    context = ResolverContext(manifests={"sampling": manifest}, sampling_unit=manifest.units[0])
    assert resolve(case, spec, resolver_context=context).reference == 0
    assert resolve({**case, "evidence": [*case["evidence"], {"id": "area-2", "type": "value", "name": "area", "value": 0.25, "ref": ["image"], "view_id": "v0"}]}, spec, resolver_context=context).reason == "argmax_tie"


class _Volume(AssetAdapter):
    kind = "volume"
    formats = frozenset({"test"})

    def load(self, asset_spec, *, source_root=None):
        return LoadedAsset(np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4), {})


def test_presentation_crop_uses_xyxy_after_slice():
    asset = AssetEntry(asset_id="volume", kind="volume", format="test", source_path="volume", file_digest="d")
    from medbenchagent.data.models import PresentationSpec

    result = _Volume().present(asset, PresentationSpec(mode="slices", axis=0, indices=[1], crop=(0.25, 0.0, 0.75, 1.0)))
    assert result.data.shape == (1, 4, 2)
    assert result.data[0, 0, 0] == 17


def test_legacy_leaf_fields_are_rejected():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        LeafTask.model_validate({
            "leaf_id": "legacy",
            "node_id": "legacy",
            "question": "old",
            "answer": "old",
        })

