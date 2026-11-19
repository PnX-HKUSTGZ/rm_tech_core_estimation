"""Manifest-to-YOLO boundary tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from tech_core_data_generator.geometry import geometry_contract, geometry_sha256

from tech_core_model_train.labels import read_yaml, write_yaml, yolo_dataset


def dataset(root: Path):
    """@brief Write a two-frame Blender test split with one negative and one partially occluded core."""
    root.mkdir()
    rng = np.random.default_rng(4)
    cv2.imwrite(str(root / "core.png"), rng.integers(0, 256, (80, 100, 3), np.uint8))
    cv2.imwrite(str(root / "background.png"), np.zeros((80, 100, 3), np.uint8))
    base = {
        "split": "test",
        "renderer": "blender",
        "width": 100,
        "height": 80,
        "bbox_xyxy": [0, 20, 80, 60],
        "pixels": [[25, 20], [50, 40], None] + [[20, 40]] * 13,
        "visibility": [2, 1, 0] + [2] * 13,
        "camera_matrix": [[110, 0, 49.5], [0, 110, 39.5], [0, 0, 1]],
        "distortion": [0.0] * 5,
        "camera_from_core": np.eye(4).tolist(),
    }
    rows = [
        dict(base, id="core", source_view="core", image="core.png", negative=False),
        dict(base, id="background", source_view="background", image="background.png", negative=True, bbox_xyxy=None),
    ]
    manifest = {"schema_version": 1, "geometry": {**geometry_contract(), "sha256": geometry_sha256()}, "records": rows}
    write_yaml(root / "manifest.yaml", manifest)
    return manifest


@pytest.mark.parametrize("task, columns", [("detect", 5), ("pose", 53)])
def test_dataset(tmp_path, task, columns):
    """@brief Labels carry normalized boxes, per-point visibility and task-specific background handling."""
    data = tmp_path / "data"
    dataset(data)
    out = tmp_path / task
    config = read_yaml(yolo_dataset(data, out, task))
    assert config["test"] == "images/test"
    values = np.loadtxt(out / "labels/test/core.txt")
    assert len(values) == columns
    np.testing.assert_allclose(values[:5], [0, 0.4, 0.5, 0.8, 0.5])
    bg = out / "labels/test/background.txt"
    if task == "pose":
        np.testing.assert_allclose(values[5:14], [0.25, 0.25, 2, 0.5, 0.5, 1, 0, 0, 0])
        assert config["kpt_shape"] == [16, 3] and not bg.exists()
    else:
        assert bg.read_text() == ""


def test_wrong_geometry(tmp_path):
    """@brief Reject a manifest whose point ordering changed under the original digest."""
    data = tmp_path / "data"
    manifest = dataset(data)
    manifest["geometry"]["names"].reverse()
    write_yaml(data / "manifest.yaml", manifest)
    with pytest.raises(ValueError, match="geometry"):
        yolo_dataset(data, tmp_path / "detect", "detect")
