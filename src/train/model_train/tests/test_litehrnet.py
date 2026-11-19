"""LiteHRNet heatmap and checkpoint contracts."""

import numpy as np
import pytest
import torch

from tech_core_model_train.litehrnet.model import LiteKeypoints, decode_heatmaps, heatmap_targets
from tech_core_model_train.litehrnet.predict import load_keypoints


def test_subpixel_heatmaps():
    """@brief Gaussian targets recover fractional crop pixels, with zero maps for missing points."""
    points = np.array([[32.25, 140.7], [0.0, 0.0], [np.nan, np.nan]])
    maps = heatmap_targets(points, [True, True, False])
    xy, peaks = decode_heatmaps(maps)
    np.testing.assert_allclose(xy[:2], points[:2], atol=0.01)
    np.testing.assert_allclose(peaks[0], np.exp(-((0.25 / 2) ** 2 + (0.7 / 2) ** 2) / (2 * 2.5**2)))
    assert peaks[2] == 0


@pytest.mark.parametrize("variant", [18, 30])
def test_output_shapes(variant):
    """@brief Both HKUST variants produce the canonical 16 heatmaps and in-frame logits."""
    torch.set_num_threads(2)
    with torch.no_grad():
        maps, logits = LiteKeypoints(variant).eval()(torch.zeros(1, 3, 256, 256))
    assert maps.shape == (1, 16, 128, 128) and logits.shape == (1, 16)


def test_mismatched_checkpoint(tmp_path):
    """@brief Geometry rejection precedes loading the network parameters."""
    path = tmp_path / "best.pt"
    torch.save({"geometry_sha256": "wrong-order"}, path)
    with pytest.raises(ValueError, match="geometry SHA-256"):
        load_keypoints(path)
