"""Public online appearance transform used by the training DataLoader."""

import numpy as np
from tech_core_data_enhancement import photometric


def test_seeded_rgb_transform_preserves_landmark_location():
    """@brief Preserve the position and colour channels of an off-centre bright marker."""
    rgb = np.zeros((32, 48, 3), np.uint8)
    rgb[4:12, 22:32, 0] = 230
    original = rgb.copy()
    output = photometric(rgb, np.random.default_rng(8))
    assert output.shape == rgb.shape and output.dtype == rgb.dtype
    assert not np.array_equal(output, rgb)
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_array_equal(output, photometric(rgb, np.random.default_rng(8)))
    np.testing.assert_array_equal(output[:, :, 0] > 50, rgb[:, :, 0] > 0)
    assert output[8, 26].argmax() == 0

