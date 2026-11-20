"""@brief 开发集选择、评估与部署导出共用的两阶段 RGB 推理。"""

from pathlib import Path

import cv2
import numpy as np
import torch

from ..labels import check_geometry
from ..prediction import Prediction, restore_points, square_crop
from ..yolo import load_yolo
from .model import LiteKeypoints, decode_heatmaps


def lite_weights(path):
    """@brief 解析 LiteHRNet 实验目录或显式检查点。

    @param[in] path LiteHRNet 实验目录或检查点。
    @return 检查点路径。
    """
    path = Path(path)
    return path / "best.pt" if path.is_dir() else path


def load_keypoints(path):
    """@brief 从已校验几何的权重中加载变体与参数。

    @param[in] path 检查点或实验目录。
    @return 加载到 CPU 的 LiteKeypoints 网络。
    """
    checkpoint = torch.load(lite_weights(path), weights_only=True, map_location="cpu")
    check_geometry(checkpoint)
    model = LiteKeypoints(checkpoint["variant"])
    model.load_state_dict(checkpoint["state_dict"])
    return model


def letterbox(rgb, side):
    """@brief 返回取值 [0,1] 的正方形浮点 RGB NCHW、缩放系数及整数左/上填充量（单位 pixel）。

    @param[in] rgb 原始 RGB8 图像。
    @param[in] side 检测器输入边长，单位 pixel。
    @return NCHW 输入、缩放系数及整数左/上填充量（单位 pixel）。
    """
    h, w = rgb.shape[:2]
    gain = side / max(h, w)
    width, height = round(w * gain), round(h * gain)
    left, top = (side - width) // 2, (side - height) // 2
    padded = np.full((side, side, 3), 114, dtype=np.uint8)
    padded[top : top + height, left : left + width] = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255, gain, np.array([left, top])


def best_box(candidates, gain, pad, width, height):
    """@brief 把得分最高的原始 [cx,cy,w,h,score] 候选解码为裁剪到原图内的 XYXY（单位 pixel）。

    @param[in] candidates 形状为 1×5×N 的原始检测器输出。
    @param[in] gain letterbox 缩放系数。
    @param[in] pad letterbox 左/上填充量，单位 pixel。
    @param[in] width 原图宽度，单位 pixel。
    @param[in] height 原图高度，单位 pixel。
    @return 上界不含的原图 XYXY（单位 pixel）与得分；漏检时为 (None, 0)。
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
    """@brief 返回取值 [0,255] 的浮点 NCHW 裁剪图及原图到裁剪图的变换（单位 pixel）。

    @param[in] rgb 原始 RGB8 图像。
    @param[in] box 原图 XYXY，单位 pixel。
    @return NCHW 输入与原图到裁剪图的变换。
    """
    crop, transform = square_crop(rgb, box)
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32), transform


def decode_prediction(box, score, heatmaps, logits, transform):
    """@brief 把热图峰值还原到原图 XY（单位 pixel），并与图像内概率合成置信度。

    @param[in] box 原图 XYXY，单位 pixel。
    @param[in] score 检测框置信度。
    @param[in] heatmaps 批量热图输出。
    @param[in] logits 批量图像内 logit。
    @param[in] transform 原图到裁剪图的像素变换。
    @return 原图中的有序关键点预测，单位 pixel。
    """
    points, peaks = decode_heatmaps(heatmaps[0])
    confidence = peaks / (1 + np.exp(-logits[0].astype(float)))
    return Prediction(box, score, restore_points(points, transform), confidence)


class DetectorLitePredictor:
    """@brief 单目标原始 YOLO11 检测，再接 LiteHRNet 关键点推理。"""

    def __init__(self, detector, keypoints, device="cpu"):
        """@brief 在指定 Torch 设备上加载两个训练好的检查点（或使用微调中的在线网络）。

        @param[in] detector 训练好的检测器检查点或实验目录。
        @param[in] keypoints LiteHRNet 检查点、实验目录，或微调期间的在线网络。
        @param[in] device Torch 设备。
        """
        yolo = load_yolo(detector, "detect")
        ## 检测器正方形输入边长，单位 pixel。
        self.imgsz = yolo.ckpt["train_args"]["imgsz"]
        ## Torch 推理设备。
        self.device = device
        ## 融合后输出原始结果的 YOLO 检测器。
        self.detector = yolo.model.fuse(verbose=False).to(device).eval()
        head = self.detector.model[-1]
        head.export, head.format = True, "onnx"
        ## 与导出端共用的关键点网络。
        self.network = keypoints if isinstance(keypoints, LiteKeypoints) else load_keypoints(keypoints)
        self.network.to(device).eval()

    @torch.inference_mode()
    def predict(self, rgb):
        """@brief 仅用 RGB 推理得分最高的检测框及 16 个原图关键点（单位 pixel）。

        @param[in] rgb 原始 RGB8 图像。
        @return 原图坐标下的预测，单位 pixel。
        """
        tensor, gain, pad = letterbox(rgb, self.imgsz)
        candidates = self.detector(torch.from_numpy(tensor).to(self.device)).cpu().numpy()
        box, score = best_box(candidates, gain, pad, rgb.shape[1], rgb.shape[0])
        if box is None:
            return Prediction.missing()
        tensor, transform = crop_input(rgb, box)
        maps, logits = self.network(torch.from_numpy(tensor).to(self.device))
        return decode_prediction(box, score, maps.cpu().numpy(), logits.cpu().numpy(), transform)
