import numpy as np

from medbenchagent.data.asset_adapters import AssetAdapter, LoadedAsset
from medbenchagent.data.v5_models import AssetEntry, PresentationSpec


class _Volume(AssetAdapter):
    kind = "volume"
    formats = frozenset({"test"})

    def load(self, asset_spec, *, source_root=None):
        return LoadedAsset(np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4), {})


def test_presentation_applies_window_crop_resize_before_slicing():
    adapter = _Volume()
    asset = AssetEntry(asset_id="v", kind="volume", format="test", source_path="v", file_digest="d")
    spec = PresentationSpec(mode="slices", axis=0, indices=[1], window=(10, 40), crop=(0, 0, 1, 1), resize=(2, 2))
    result = adapter.present(asset, spec)
    assert result.data.shape == (1, 2, 2)
    assert result.data.dtype == np.uint8
