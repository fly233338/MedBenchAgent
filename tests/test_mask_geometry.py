import numpy as np

from medbenchagent.data.mask_geometry import extract_mask_features, geometry_digest


def test_volume_geometry_uses_native_shape_axis_two_and_is_repeatable():
    mask = np.zeros((4, 5, 3), dtype=np.uint8)
    mask[1:3, 2:4, 1] = 1
    features = extract_mask_features(mask, source_digest="mask", foreground=[1])
    repeated = extract_mask_features(mask.copy(), source_digest="mask", foreground=[1])

    assert features.shape == [4, 5, 3]
    assert features.component_count == 1
    assert [(plane.axis, plane.index) for plane in features.planes] == [(2, 1)]
    assert features.bbox == [0.25, 0.4, 0.75, 0.8]
    assert features.centroid == [0.5, 0.6]
    assert geometry_digest(features) == geometry_digest(repeated)


def test_unknown_spacing_is_not_materialized_as_physical_area():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    features = extract_mask_features(mask, source_digest="mask", foreground=[1], spacing=[2, 3])
    assert features.spacing is None
    assert features.physical_area is None
