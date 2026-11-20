"""关键点预测与图像坐标契约。"""

from unittest.mock import patch

import numpy as np
import torch
from ultralytics.engine.results import Results

from tech_core_model_train.litehrnet.predict import best_box, letterbox
from tech_core_model_train.prediction import restore_points, square_crop
from tech_core_model_train.yolo import YoloPosePredictor


def test_padded_crop_round_trip():
    """@brief 贴近左边缘的检测框保持真实中心，并使用黑色填充。"""
    image = np.full((85, 123, 3), 255, np.uint8)
    crop, transform = square_crop(image, [0.0, 11.0, 30.0, 81.0])
    points = np.array([[15.0, 46.0], [0.0, 11.0], [29.2, 80.2]])
    mapped = points @ transform[:2, :2].T + transform[:2, 2]
    np.testing.assert_allclose(mapped[0], 127.5)
    np.testing.assert_allclose(restore_points(mapped, transform), points, atol=1e-12)
    assert crop.shape == (256, 256, 3) and not crop[:, 0].any()


def test_odd_letterbox_padding_and_top_candidate():
    """@brief 在总填充量为奇数的非正方形画面上还原 XYXY。"""
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


def test_yolo_pose_discards_points_clipped_to_any_edge():
    """@brief 四条边上的夹取坐标无效，靠近边界的画内预测保留置信度。"""
    rgb = np.zeros((40, 60, 3), np.uint8)
    points = torch.tensor([[[0, 15, 0.9], [25, 0, 0.9], [60, 15, 0.9], [25, 40, 0.9], [0.1, 39.9, 0.7]]])
    boxes = torch.tensor([[0, 0, 60, 40, 0.9, 0]])
    result = Results(rgb, path="frame", names={0: "core"}, boxes=boxes, keypoints=points)
    with patch("tech_core_model_train.yolo.load_yolo") as load:
        load.return_value.ckpt = {"train_args": {"imgsz": 320}}
        load.return_value.predict.return_value = [result]
        prediction = YoloPosePredictor("unused.pt").predict(rgb)
    np.testing.assert_allclose(prediction.confidence, [0, 0, 0, 0, 0.7])
    np.testing.assert_array_equal(prediction.valid(60, 40, 0.25), [False, False, False, False, True])
