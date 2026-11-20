"""@brief 原图坐标下的预测结果与仿射裁剪变换。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Prediction:
    """@brief 一帧 RGB 中得分最高的检测框及有序关键点预测。"""

    bbox_xyxy: np.ndarray | None  ##< 上界不含的原图包围框，单位 pixel；漏检时为 None。
    score: float  ##< 检测器置信度。
    pixels: np.ndarray  ##< 原图 XY，单位 pixel，按几何顺序排列。
    confidence: np.ndarray  ##< 逐点置信度，取值 [0, 1]。

    @classmethod
    def missing(cls):
        """@brief 返回含 16 个缺失点的空检测结果。

        @return 空检测框、NaN 坐标与零置信度。
        """
        return cls(None, 0.0, np.full((16, 2), np.nan), np.zeros(16))

    def valid(self, width, height, threshold):
        """@brief 标记位于图像内且达到冻结置信度门限的观测。

        @param[in] width 原图宽度，单位 pixel。
        @param[in] height 原图高度，单位 pixel。
        @param[in] threshold 置信度下限。
        @return 有序的 16 元素掩码。
        """
        return (
            (self.confidence >= threshold)
            & (self.pixels >= 0).all(axis=1)
            & (self.pixels < [width, height]).all(axis=1)
        )


def square_crop(rgb, box):
    """@brief 围绕检测框截取 1.25 倍正方形区域，图像外部分保留填充。

    @param[in] rgb 原始 RGB uint8 图像。
    @param[in] box 面积为正、上界不含的 XYXY 包围框（原图，单位 pixel）。
    @return 256×256 RGB 裁剪图及原图到裁剪图的齐次像素变换。
    """
    box = np.asarray(box, dtype=float)
    center = (box[:2] + box[2:]) / 2
    scale = 256 / (1.25 * np.max(box[2:] - box[:2]))
    transform = np.array([[scale, 0, 127.5 - scale * center[0]], [0, scale, 127.5 - scale * center[1]], [0, 0, 1]])
    return cv2.warpAffine(rgb, transform[:2], (256, 256), flags=cv2.INTER_LINEAR), transform


def restore_points(points, transform):
    """@brief 对原图到裁剪图的仿射变换求逆，把裁剪图 XY 点还原为原图 XY（单位 pixel）。

    @param[in] points 裁剪图 XY 坐标，单位 pixel。
    @param[in] transform square_crop() 返回的矩阵。
    @return 原图 XY 坐标，单位 pixel。
    """
    return np.c_[points, np.ones(len(points))] @ np.linalg.inv(transform)[:2].T
