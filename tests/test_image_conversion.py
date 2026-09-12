import numpy as np
import pytest
from PIL import Image

from medbenchagent.data.image_conversion import (
    ConversionError, UnsupportedMedicalImage, convert_image, inspect_image, write_image,
)


@pytest.mark.parametrize("encoding", ["PNG", "JPEG"])
def test_png_jpeg_copy_is_content_based_and_byte_identical(tmp_path, encoding):
    source = tmp_path / "misleading.nii"
    Image.new("RGB", (3, 2), (90, 20, 70)).save(source, format=encoding)
    target = tmp_path / "output.bin"
    write_image(source, target)
    assert target.read_bytes() == source.read_bytes()


def test_uint16_tiff_retains_bit_depth_and_pixel_values(tmp_path):
    values = np.array([[0, 1, 400], [500, 5000, 65535]], dtype=np.uint16)
    source = tmp_path / "source.tif"
    Image.fromarray(values).save(source)
    target = tmp_path / "image.png"
    write_image(source, target)
    with Image.open(target) as image:
        assert image.size == (3, 2)
        np.testing.assert_array_equal(np.asarray(image), values)


def test_float_pixels_need_explicit_window_and_multiframe_is_rejected(tmp_path):
    source = tmp_path / "float.tif"
    Image.fromarray(np.array([[0, 5], [10, 20]], dtype=np.float32)).save(source)
    assert not inspect_image(source)["preserve_supported"]
    with pytest.raises(ConversionError, match="explicit user decision"):
        convert_image(source)
    rendered, _, _ = convert_image(source, {"encoding": "window", "low": 0, "high": 20})
    np.testing.assert_array_equal(np.asarray(rendered), [[0, 64], [128, 255]])
    Image.new("L", (2, 2)).save(source, save_all=True, append_images=[Image.new("L", (2, 2))])
    with pytest.raises(UnsupportedMedicalImage, match="Multiframe"):
        inspect_image(source)
