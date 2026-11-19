"""@brief HKUST LiteHRNet-18/30 architecture adapted from 12 landmarks to the confirmed 16."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .vendor.litehrnet import LiteHRNet


class LiteKeypoints(nn.Module):
    """@brief HKUST LiteHRNet with landmark heatmaps and pooled in-frame logits."""

    def __init__(self, variant: int = 18):
        """@brief Construct the requested 16-channel LiteHRNet variant.

        @param[in] variant Depth, 18 or 30.
        """
        super().__init__()
        ## Architecture depth (18 or 30).
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
        ## COCO-compatible feature extractor.
        self.backbone = LiteHRNet(extra=extra, in_channels=3)
        channels = 40 if variant == 18 else 256
        ## Upsampling head for the 16 heatmaps.
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
        # HKUST visibility_head.py applies this head to the backbone output, before deconvolution.
        ## One in-frame logit per landmark.
        self.visibility_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(40, 16))
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, rgb):
        """@brief Convert float RGB in [0,255] to Nx16x128x128 heatmaps and Nx16 in-frame logits.

        @param[in] rgb NCHW batch of 256×256 RGB crops on the 0–255 scale.
        @return Raw 128×128 heatmaps and in-frame logits in geometry order.
        """
        features = self.backbone((rgb - self.mean) / self.std)[0]
        return self.heatmap_head(features), self.visibility_head(features)

    def load_pretrained_backbone(self, path: str | Path) -> None:
        """@brief Load compatible COCO backbone weights, never reuse the unrelated human landmark head.

        @param[in] path Trusted upstream checkpoint downloaded from OpenMMLab.
        """
        raw = torch.load(path, map_location="cpu", weights_only=False)
        weights = raw.get("state_dict", raw)
        weights = {k.removeprefix("backbone."): v for k, v in weights.items() if k.startswith("backbone.")}
        self.backbone.load_state_dict(weights, strict=True)


def heatmap_targets(pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """@brief Encode the training targets for a crop.

    @param[in] pixels XY centres in the 256×256 crop (pixel).
    @param[in] valid Per-point in-frame mask.
    @return 16×128×128 continuous-centre Gaussians, σ = 2.5 heatmap pixels.
    """
    y, x = np.mgrid[:128, :128]
    centers = np.asarray(pixels, dtype=float) / 2
    safe = np.nan_to_num(centers, nan=-1e6)
    maps = np.exp(-((x - safe[:, 0, None, None]) ** 2 + (y - safe[:, 1, None, None]) ** 2) / (2 * 2.5**2))
    maps[~np.asarray(valid, dtype=bool)] = 0
    return maps.astype(np.float32)


def decode_heatmaps(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """@brief Refine each raw maximum by a local quadratic fit to the logarithm of the smoothed heatmap.

    @param[in] heatmaps 16×128×128 raw network outputs.
    @return XY in 256×256 crop coordinates (pixel) and peaks clipped to [0,1].
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
