"""@brief Two-stage RGB inference shared by development selection, evaluation and deployment export."""

from pathlib import Path

import cv2
import numpy as np
import torch

from ..labels import check_geometry
from ..prediction import Prediction, restore_points, square_crop
from ..yolo import load_yolo
from .model import LiteKeypoints, decode_heatmaps


def lite_weights(path):
    """@brief Resolve a LiteHRNet experiment or explicit checkpoint.

    @param[in] path LiteHRNet experiment directory or checkpoint.
    @return Checkpoint path.
    """
    path = Path(path)
    return path / "best.pt" if path.is_dir() else path


def load_keypoints(path):
    """@brief Load the variant and parameters from geometry-checked weights.

    @param[in] path Checkpoint or experiment directory.
    @return Loaded CPU LiteKeypoints network.
    """
    checkpoint = torch.load(lite_weights(path), weights_only=True, map_location="cpu")
    check_geometry(checkpoint)
    model = LiteKeypoints(checkpoint["variant"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


def letterbox(rgb, side):
    """@brief Return square RGB float NCHW in [0,1], resize gain and integer left/top padding (pixel).

    @param[in] rgb Original RGB8 image.
    @param[in] side Detector input side (pixel).
    @return NCHW input, resize gain and integer left/top padding (pixel).
    """
    h, w = rgb.shape[:2]
    gain = side / max(h, w)
    width, height = round(w * gain), round(h * gain)
    left, top = (side - width) // 2, (side - height) // 2
    padded = np.full((side, side, 3), 114, dtype=np.uint8)
    padded[top : top + height, left : left + width] = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255, gain, np.array([left, top])


def best_box(candidates, gain, pad, width, height):
    """@brief Decode the highest raw [cx,cy,w,h,score] candidate into clipped original XYXY (pixel).

    @param[in] candidates Raw detector output of shape 1×5×N.
    @param[in] gain Letterbox resize gain.
    @param[in] pad Left/top letterbox padding (pixel).
    @param[in] width Original image width (pixel).
    @param[in] height Original image height (pixel).
    @return Exclusive original-image XYXY (pixel) and score, or (None, 0) on a miss.
    """
    x, y, w, h, score = candidates[0, :, candidates[0, 4].argmax()]
    if score < 0.25:
        return None, 0.0
    center = np.array([x, y])
    half = np.array([w, h]) / 2
    box = np.clip((np.array([center - half, center + half]) - pad) / gain, 0, [width, height])
    if np.any(box[1] <= box[0]):
        return None, 0.0
    return box.ravel(), float(score)


def crop_input(rgb, box):
    """@brief Return float NCHW crop in [0,255] and original-image to crop transform (pixel).

    @param[in] rgb Original RGB8 image.
    @param[in] box Original-image XYXY (pixel).
    @return NCHW input and image-to-crop transform.
    """
    crop, transform = square_crop(rgb, box)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32), transform


def decode_prediction(box, score, heatmaps, logits, transform):
    """@brief Restore heatmap peaks to original XY (pixel) and combine peaks with in-frame probabilities.

    @param[in] box Original-image XYXY (pixel).
    @param[in] score Box confidence.
    @param[in] heatmaps Batched heatmap output.
    @param[in] logits Batched in-frame logits.
    @param[in] transform Image-to-crop pixel transform.
    @return Ordered landmark prediction in the original image (pixel).
    """
    points, peaks = decode_heatmaps(heatmaps[0])
    confidence = peaks / (1 + np.exp(-logits[0].astype(float)))
    return Prediction(box, score, restore_points(points, transform), confidence)


class DetectorLitePredictor:
    """@brief Single-object raw YOLO11 detection followed by LiteHRNet landmark inference."""

    def __init__(self, detector, keypoints, device="cpu"):
        """@brief Load the two trained checkpoints on the given Torch device (or use a live fine-tuning network).

        @param[in] detector Trained detector checkpoint or experiment.
        @param[in] keypoints LiteHRNet checkpoint, experiment or live network during fine-tuning.
        @param[in] device Torch device.
        """
        yolo = load_yolo(detector, "detect")
        ## Square detector input side (pixel).
        self.imgsz = yolo.ckpt["train_args"]["imgsz"]
        ## Torch inference device.
        self.device = device
        ## Fused YOLO detector with raw outputs.
        self.detector = yolo.model.fuse(verbose=False).to(device).eval()
        head = self.detector.model[-1]
        head.export, head.format = True, "onnx"
        ## Landmark network shared with the exporter.
        self.network = keypoints if isinstance(keypoints, LiteKeypoints) else load_keypoints(keypoints)
        self.network.to(device).eval()

    @torch.inference_mode()
    def predict(self, rgb):
        """@brief Infer the highest-scoring box and 16 original-image landmarks (pixel) using only RGB.

        @param[in] rgb Original RGB8 image.
        @return Prediction in original-image coordinates (pixel).
        """
        tensor, gain, pad = letterbox(rgb, self.imgsz)
        candidates = self.detector(torch.from_numpy(tensor).to(self.device)).cpu().numpy()
        box, score = best_box(candidates, gain, pad, rgb.shape[1], rgb.shape[0])
        if box is None:
            return Prediction.missing()
        tensor, transform = crop_input(rgb, box)
        maps, logits = self.network(torch.from_numpy(tensor).to(self.device))
        return decode_prediction(box, score, maps.cpu().numpy(), logits.cpu().numpy(), transform)
