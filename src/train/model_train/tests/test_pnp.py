"""Distorted planar and spatial reference PnP, with missing landmarks and an outlier."""

import cv2
import numpy as np
import pytest

from tech_core_model_train.pnp import KeypointModel, estimate


@pytest.mark.parametrize("planar", [False, True])
def test_consensus_with_distortion(planar):
    """@brief Recover the known core pose despite a mismatched pixel and masked observations."""
    model = KeypointModel()
    camera = np.array([[520.0, 0.0, 319.5], [0.0, 510.0, 239.5], [0.0, 0.0, 1.0]])
    dist = np.array([-0.12, 0.035, 0.001, -0.002, 0.004])
    rotation, translation = np.array([0.2, -1.2, 0.1]), np.array([0.03, -0.02, 0.8])
    pixels = cv2.projectPoints(model.points, rotation, translation, camera, dist)[0].reshape(-1, 2)
    valid = np.ones(16, bool)
    if planar:
        valid = model.points[:, 0] == -0.1
    pixels[7] += [52.0, -40.0]
    pixels[8] = np.nan
    result = estimate(model, pixels, valid, camera, dist)
    np.testing.assert_allclose(result[:3, :3], cv2.Rodrigues(rotation)[0], atol=1e-7)
    np.testing.assert_allclose(result[:3, 3], translation, atol=1e-8)


def test_too_few_points():
    """@brief Three usable landmarks cannot define a reference pose."""
    assert estimate(KeypointModel(), np.ones((16, 2)), np.arange(16) < 3, np.eye(3)) is None
