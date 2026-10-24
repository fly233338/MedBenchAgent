from medbenchagent.data.models import AssetEntry, PresentationSpec
from medbenchagent.data.sources import digest_file
from medbenchagent.data.readers import RasterReader

import numpy as np
import pytest
from PIL import Image

from medbenchagent.data.asset_adapters import (
    ConversionError,
    UnsupportedMedicalImage,
    materialize_image,
    detect_format,
)


@pytest.mark.parametrize("encoding", ["PNG", "JPEG"])
def test_explicit_raster_conversion_writes_real_png(tmp_path, encoding):
    source = tmp_path / ("source.png" if encoding == "PNG" else "source.jpg")
    Image.new("RGB", (3, 2), (90, 20, 70)).save(source, format=encoding)
    target = tmp_path / "output.png"
    asset = AssetEntry(
        asset_id="source",
        kind="image",
        format=detect_format(source).split(":")[-1],
        source_path=source.name,
        file_digest=digest_file(source),
    )
    materialize_image(asset, PresentationSpec(), target, source_root=tmp_path)
    with Image.open(target) as rendered, Image.open(source) as original:
        assert rendered.format == "PNG"
        np.testing.assert_array_equal(np.asarray(rendered), np.asarray(original))


def test_uint16_tiff_retains_bit_depth_and_pixel_values(tmp_path):
    values = np.array([[0, 1, 400], [500, 5000, 65535]], dtype=np.uint16)
    source = tmp_path / "source.tif"
    Image.fromarray(values).save(source)
    target = tmp_path / "image.png"
    asset = AssetEntry(
        asset_id="source",
        kind="image",
        format=detect_format(source).split(":")[-1],
        source_path=source.name,
        file_digest=digest_file(source),
    )
    materialize_image(asset, PresentationSpec(), target, source_root=tmp_path)
    with Image.open(target) as image:
        assert image.size == (3, 2)
        np.testing.assert_array_equal(np.asarray(image), values)


def test_float_pixels_need_explicit_window_and_multiframe_is_rejected(tmp_path):
    source = tmp_path / "float.tif"
    Image.fromarray(np.array([[0, 5], [10, 20]], dtype=np.float32)).save(source)
    asset = AssetEntry(
        asset_id="source",
        kind="image",
        format="tiff",
        source_path=source.name,
        file_digest=digest_file(source),
    )
    target = tmp_path / "output.png"
    with pytest.raises(ConversionError, match="explicit user decision"):
        materialize_image(asset, PresentationSpec(), target, source_root=tmp_path)
    materialize_image(asset, PresentationSpec(window=(0, 20)), target, source_root=tmp_path)
    with Image.open(target) as rendered:
        np.testing.assert_array_equal(np.asarray(rendered), [[0, 64], [128, 255]])
    Image.new("L", (2, 2)).save(source, save_all=True, append_images=[Image.new("L", (2, 2))])
    with pytest.raises(UnsupportedMedicalImage, match="Multiframe"):
        RasterReader().read_pixels(source)
