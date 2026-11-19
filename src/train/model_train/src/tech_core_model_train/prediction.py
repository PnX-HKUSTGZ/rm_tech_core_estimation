"""@brief Predictions and affine crop transforms in original image coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Prediction:
    """@brief Highest scoring detection and ordered landmark predictions for one RGB frame."""

    bbox_xyxy: np.ndarray | None  ##< Exclusive original-image bounds (pixel), None on a miss.
    score: float  ##< Detector confidence.
    pixels: np.ndarray  ##< Original-image XY (pixel), in geometry order.
    confidence: np.ndarray  ##< Per-point confidence in [0, 1].

    @classmethod
    def missing(cls):
        """@brief Return an empty detection with 16 missing points.

        @return Empty box, NaN coordinates and zero confidence.
        """
        return cls(None, 0.0, np.full((16, 2), np.nan), np.zeros(16))

    def valid(self, width, height, threshold):
        """@brief Mark in-frame observations meeting the frozen confidence threshold.

        @param[in] width Original image width (pixel).
        @param[in] height Original image height (pixel).
        @param[in] threshold Minimum confidence.
        @return Ordered 16-element mask.
        """
        return (
            (self.confidence >= threshold)
            & (self.pixels >= 0).all(axis=1)
            & (self.pixels < [width, height]).all(axis=1)
        )


def square_crop(rgb, box):
    """@brief Warp a 1.25× square around a box, preserving out-of-image padding.

    @param[in] rgb Original RGB uint8 image.
    @param[in] box Positive exclusive XYXY bounds (original image, pixel).
    @return 256×256 RGB crop and homogeneous image-to-crop pixel transform.
    """
    box = np.asarray(box, dtype=float)
    center = (box[:2] + box[2:]) / 2
    scale = 256 / (1.25 * np.max(box[2:] - box[:2]))
    transform = np.array([[scale, 0, 127.5 - scale * center[0]], [0, scale, 127.5 - scale * center[1]], [0, 0, 1]])
    return cv2.warpAffine(rgb, transform[:2], (256, 256), flags=cv2.INTER_LINEAR), transform


def restore_points(points, transform):
    """@brief Invert an image-to-crop affine transform for crop XY points and return original image XY (pixel).

    @param[in] points Crop XY coordinates (pixel).
    @param[in] transform Matrix returned by square_crop().
    @return Original-image XY coordinates (pixel).
    """
    return np.c_[points, np.ones(len(points))] @ np.linalg.inv(transform)[:2].T
