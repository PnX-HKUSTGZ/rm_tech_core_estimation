"""Geometry fingerprint, optical convention and sampling contract."""

import numpy as np
import pytest
from tech_core_data_generator import CameraConfig, Viewpoint, geometry_contract, geometry_sha256
from tech_core_data_generator.camera import project_points, sample_random
from tech_core_data_generator.geometry import MODEL_DIR, read_obj


def test_geometry_matches_rule_mesh_and_published_fingerprint():
    """@brief Pin the packaged geometry to the physical model, rule lengths and digest."""
    contract = geometry_contract()
    assert contract["profile"] == "shell5_rule_lengths_model_reference_v2"
    assert len(set(contract["names"])) == 16
    assert contract["groups"] == ["shell"] * 5 + ["marker"] * 10 + ["pillar"]
    left, top, right, lower_left, lower_right = np.array(contract["points_m"][:5])
    np.testing.assert_allclose(np.linalg.norm(right - top), 0.098, atol=1e-12)
    np.testing.assert_allclose(right - lower_right, [0, 0, 0.074], atol=1e-12)
    np.testing.assert_allclose([left, lower_left], [right, lower_right] * np.array([1, -1, 1]), atol=1e-12)
    vertices, _ = read_obj(MODEL_DIR / "meshes/shell.obj")
    for point in (left, top, right, lower_left, lower_right):
        assert np.linalg.norm(vertices - point, axis=1).min() < 2e-7
    contract["names"].reverse()
    assert geometry_contract()["names"][0] == "shell_seam_left"
    assert geometry_sha256() == "66e7cad483b1cd9aca0ceebeeb06652a99791f16689438a93c6bd6a2e76ac34a"


def test_front_camera_projection_and_clipping():
    """@brief Check an analytic projection along the core X axis."""
    camera = CameraConfig(160, 120, fovy=np.pi / 2)
    view = Viewpoint(0.8, target=(0, 0, 0))
    pixels, depths, inside = project_points(
        np.array([[0.2, 0.1, 0.2], [1, 0, 0], [0, 2, 0]]), camera, view.camera_from_core
    )
    np.testing.assert_allclose(pixels[0], [89.5, 39.5])
    np.testing.assert_allclose(depths, [0.6, -0.2, 0.8])
    assert np.isnan(pixels[1]).all()
    assert inside.tolist() == [True, False, False]


def test_reject_invalid_camera_settings():
    """@brief Exercise invalid projection geometry exposed through the preview options."""
    with pytest.raises(ValueError, match="image size"):
        CameraConfig(width=0)
    with pytest.raises(ValueError, match="radius"):
        Viewpoint(r=-0.2)
    with pytest.raises(ValueError, match="target"):
        _ = Viewpoint(0.8, target=(0.8, 0.0, 0.0)).camera_from_core


@pytest.mark.parametrize(
    "name,low,high",
    [
        ("regular", [0.45, -0.96, -0.35], [1, 0.96, 0.61]),
        ("stress", [0.2, -1.57, -1.05], [1, 1.57, 1.05]),
        ("training", [0.25, -1.4, -0.96], [1, 1.4, 0.96]),
    ],
)
def test_seeded_distributions(name, low, high):
    """@brief Measure support and mixture weights for a fixed random stream."""
    rng = np.random.default_rng(41)
    views = [sample_random(rng, name) for _ in range(2000)]
    again = np.random.default_rng(41)
    assert views[:3] == [sample_random(again, name) for _ in range(3)]
    samples = np.array([(view.r, view.theta, view.alpha) for view in views])
    assert np.all(samples >= low) and np.all(samples <= high)
    np.testing.assert_allclose(samples.min(axis=0), low, atol=0.025)
    np.testing.assert_allclose(samples.max(axis=0), high, atol=0.025)
    if name == "training":
        # Only the 15% wide component samples radii below 0.4 m; its fraction there is 0.15 / 0.75.
        assert np.mean(samples[:, 0] < 0.4) == pytest.approx(0.15 * 0.15 / 0.75, abs=0.012)
