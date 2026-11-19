"""Crop and letterbox coordinate contracts."""

import numpy as np

from tech_core_model_train.litehrnet.predict import best_box, letterbox
from tech_core_model_train.prediction import restore_points, square_crop


def test_padded_crop_round_trip():
    """@brief A box against the left edge keeps its true centre and uses black padding."""
    image = np.full((85, 123, 3), 255, np.uint8)
    crop, transform = square_crop(image, [0.0, 11.0, 30.0, 81.0])
    points = np.array([[15.0, 46.0], [0.0, 11.0], [29.2, 80.2]])
    mapped = points @ transform[:2, :2].T + transform[:2, 2]
    np.testing.assert_allclose(mapped[0], 127.5)
    np.testing.assert_allclose(restore_points(mapped, transform), points, atol=1e-12)
    assert crop.shape == (256, 256, 3) and not crop[:, 0].any()


def test_odd_letterbox_padding_and_top_candidate():
    """@brief Restore XYXY on a non-square frame with odd total padding."""
    image = np.zeros((51, 100, 3), np.uint8)
    image[..., 0] = 255
    tensor, gain, pad = letterbox(image, 320)
    assert tensor.shape == (1, 3, 320, 320)
    np.testing.assert_allclose(tensor[0, :, 77, 0], 114 / 255)
    np.testing.assert_allclose(tensor[0, :, 78, 0], [1, 0, 0])
    candidates = np.array([[[160, 32], [158, 110], [128, 25.6], [64, 32], [0.1, 0.9]]])
    box, score = best_box(candidates, gain, pad, 100, 51)
    np.testing.assert_allclose(box, [6, 5, 14, 15])
    assert score == 0.9
