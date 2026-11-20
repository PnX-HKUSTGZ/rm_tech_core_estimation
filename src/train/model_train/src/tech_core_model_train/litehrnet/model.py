"""@brief HKUST LiteHRNet-18/30 架构，由 12 个关键点改为确定的 16 个。"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .vendor.litehrnet import LiteHRNet


class LiteKeypoints(nn.Module):
    """@brief 输出关键点热图与池化后图像内 logit 的 HKUST LiteHRNet。"""

    def __init__(self, variant: int = 18):
        """@brief 构造指定的 16 通道 LiteHRNet 变体。

        @param[in] variant 网络深度，18 或 30。
        """
        super().__init__()
        ## 网络深度（18 或 30）。
        self.variant = variant
        extra = {
            "stem": {"stem_channels": 32, "out_channels": 32, "expand_ratio": 1},
            "num_stages": 3,
            "stages_spec": {
                "num_modules": (2, 4, 2) if variant == 18 else (3, 8, 3),
                "num_branches": (2, 3, 4),
                "num_blocks": (2, 2, 2),
                "module_type": ("LITE",) * 3,
                "with_fuse": (True,) * 3,
                "reduce_ratios": (8,) * 3,
                "num_channels": ((40, 80), (40, 80, 160), (40, 80, 160, 320)),
            },
            "with_head": True,
        }
        ## 兼容 COCO 权重的特征提取器。
        self.backbone = LiteHRNet(extra=extra, in_channels=3)
        channels = 40 if variant == 18 else 256
        ## 生成 16 张热图的上采样头。
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(40, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 16, 1),
        )
        for layer in self.heatmap_head.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight, std=0.001)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # HKUST 的 visibility_head.py 把该头接在主干输出上，位于反卷积之前。
        ## 每个关键点一个图像内 logit。
        self.visibility_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(40, 16))
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, rgb):
        """@brief 把取值 [0,255] 的浮点 RGB 转为 Nx16x128x128 热图与 Nx16 图像内 logit。

        @param[in] rgb 256×256 RGB 裁剪图的 NCHW 批次，0–255 刻度。
        @return 按几何顺序排列的原始 128×128 热图与图像内 logit。
        """
        features = self.backbone((rgb - self.mean) / self.std)[0]
        return self.heatmap_head(features), self.visibility_head(features)

    def load_pretrained_backbone(self, path: str | Path) -> None:
        """@brief 加载兼容的 COCO 主干权重，不复用无关的人体关键点头。

        @param[in] path 从 OpenMMLab 下载的可信上游检查点。
        """
        raw = torch.load(path, map_location="cpu", weights_only=False)
        weights = raw.get("state_dict", raw)
        weights = {k.removeprefix("backbone."): v for k, v in weights.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(weights, strict=True)


def heatmap_targets(pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """@brief 为一张裁剪图编码训练目标。

    @param[in] pixels 256×256 裁剪图中的 XY 中心，单位 pixel。
    @param[in] valid 逐点图像内掩码。
    @return 16×128×128 连续中心高斯图，σ = 2.5 个热图像素。
    """
    y, x = np.mgrid[:128, :128]
    centers = np.asarray(pixels, dtype=float) / 2
    safe = np.nan_to_num(centers, nan=-1e6)
    maps = np.exp(-((x - safe[:, 0, None, None]) ** 2 + (y - safe[:, 1, None, None]) ** 2) / (2 * 2.5**2))
    maps[~np.asarray(valid, dtype=bool)] = 0
    return maps.astype(np.float32)


def decode_heatmaps(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """@brief 对平滑后热图取对数做局部二次拟合，细化每个原始极大值。

    @param[in] heatmaps 16×128×128 原始网络输出。
    @return 256×256 裁剪图坐标下的 XY（单位 pixel）及裁剪到 [0,1] 的峰值。
    """
    points, scores = [], []
    for raw in heatmaps:
        y, x = np.unravel_index(np.argmax(raw), raw.shape)
        peak = float(raw[y, x])
        delta = np.zeros(2)
        if 1 <= x < raw.shape[1] - 1 and 1 <= y < raw.shape[0] - 1:
            log = np.log(np.clip(cv2.GaussianBlur(raw, (3, 3), 0), 1e-10, None))
            gradient = np.array([(log[y, x + 1] - log[y, x - 1]) / 2, (log[y + 1, x] - log[y - 1, x]) / 2])
            cross = (log[y + 1, x + 1] - log[y + 1, x - 1] - log[y - 1, x + 1] + log[y - 1, x - 1]) / 4
            hessian = np.array(
                [
                    [log[y, x + 1] + log[y, x - 1] - 2 * log[y, x], cross],
                    [cross, log[y + 1, x] + log[y - 1, x] - 2 * log[y, x]],
                ]
            )
            if np.linalg.det(hessian) > 1e-8 and np.trace(hessian) < 0:
                candidate = -np.linalg.solve(hessian, gradient)
                if np.max(np.abs(candidate)) < 1.5:
                    delta = candidate
        points.append((np.array([x, y]) + delta) * 2)
        scores.append(np.clip(peak, 0, 1))
    return np.asarray(points), np.asarray(scores)
