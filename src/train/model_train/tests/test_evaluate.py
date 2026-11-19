"""Failure-inclusive held-out metrics and threshold selection boundaries."""

import cv2
import numpy as np
import pytest

from tech_core_model_train.evaluate import aggregate, calibration_rows, score_prediction
from tech_core_model_train.pnp import KeypointModel
from tech_core_model_train.prediction import Prediction


def test_pose_denominator_and_background_precision():
    """@brief A hit, miss, failed solve, inaccurate solve and background FP use distinct denominators."""
    geometry = KeypointModel()
    camera = np.array([[240.0, 0.0, 160.0], [0.0, 240.0, 120.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[2, 3] = 1.0
    xy = cv2.projectPoints(geometry.points, np.zeros(3), pose[:3, 3], camera, None)[0].reshape(-1, 2)
    row = {
        "id": "target",
        "split": "test",
        "renderer": "blender",
        "image": "a.png",
        "width": 320,
        "height": 240,
        "negative": False,
        "bbox_xyxy": [100.0, 60.0, 190.0, 220.0],
        "camera_matrix": camera,
        "distortion": [],
        "camera_from_core": pose,
        "pixels": xy,
        "visibility": [2] * 16,
    }
    pred = Prediction(np.array(row["bbox_xyxy"]), 0.9, xy, np.ones(16))
    unsolved = Prediction(pred.bbox_xyxy, 0.9, xy, np.zeros(16))
    wrong = pose.copy()
    wrong[0, 3] += 0.02
    frames = [
        score_prediction(r, p, 0.25, geometry)
        for r, p in (
            (row, pred),
            (row, Prediction.missing()),
            (row, unsolved),
            (dict(row, camera_from_core=wrong), pred),
            (dict(row, negative=True, bbox_xyxy=None), pred),
        )
    ]
    result = aggregate(frames)
    assert result["positive_frames"] == 4 and result["background_false_positives"] == 1
    assert result["detection_precision"] == result["detection_recall"] == 0.75
    assert result["solve_rate"] == result["pck5"] == 0.5 and result["accurate_rate"] == 0.25
    np.testing.assert_allclose(result["accuracy_wilson95"], [0.045587, 0.699358], atol=1e-6)
    assert result["failures"] == {"no_box": 1, "box_unsolved": 1, "solved_out_of_tolerance": 1}


def test_calibration_domains():
    """@brief Test and MuJoCo acceptance rows cannot implicitly become validation samples."""
    rows = [{"renderer": "blender", "split": "test"}, {"renderer": "mujoco", "split": "regular"}]
    with pytest.raises(ValueError, match="calibration"):
        calibration_rows({"records": rows}, False)
    assert calibration_rows({"records": rows}, True) == rows[1:]
