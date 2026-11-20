"""LiteHRNet 热图与检查点契约。"""

import numpy as np
import pytest
import torch

from tech_core_model_train.litehrnet.model import LiteKeypoints, decode_heatmaps, heatmap_targets
from tech_core_model_train.litehrnet.predict import load_keypoints


def test_subpixel_heatmaps():
    """@brief 高斯目标能还原裁剪图中的小数像素坐标，缺失点对应全零热图。"""
    points = np.array([[32.25, 140.7], [0.0, 0.0], [np.nan, np.nan]])
    maps = heatmap_targets(points, [True, True, False])
    xy, peaks = decode_heatmaps(maps)
    np.testing.assert_allclose(xy[:2], points[:2], atol=0.01)
    np.testing.assert_allclose(peaks[0], np.exp(-((0.25 / 2) ** 2 + (0.7 / 2) ** 2) / (2 * 2.5**2)))
    assert peaks[2] == 0


@pytest.mark.parametrize("variant", [18, 30])
def test_output_shapes(variant):
    """@brief 两个 HKUST 变体都输出规范的 16 张热图与图像内 logit。"""
    torch.set_num_threads(2)
    with torch.no_grad():
        maps, logits = LiteKeypoints(variant).eval()(torch.zeros(1, 3, 256, 256))
    assert maps.shape == (1, 16, 128, 128) and logits.shape == (1, 16)


def test_mismatched_checkpoint(tmp_path):
    """@brief 在加载网络参数之前先进行几何校验并拒绝不符者。"""
    path = tmp_path / "best.pt"
    torch.save({"geometry_sha256": "wrong-order"}, path)
    with pytest.raises(ValueError, match="geometry SHA-256"):
        load_keypoints(path)
