"""@brief 预训练 YOLO11 检测器与直接回归 16 点的 pose 基线。"""

from pathlib import Path

import numpy as np
import torch
from tech_core_data_generator.dataset import require_empty
from ultralytics import YOLO

from .labels import check_geometry, experiment, pretrained_file, sha256, write_yaml, yolo_dataset
from .prediction import Prediction


def yolo_weights(path):
    """@brief 解析 YOLO 实验目录或显式检查点。

    @param[in] path 检测器或 pose 的实验目录或检查点。
    @return 检查点路径。
    """
    path = Path(path)
    return path / "weights/best.pt" if path.is_dir() else path


def load_yolo(path, task):
    """@brief 加载训练好的检查点，并校验规范关键点标识与任务类型。

    @param[in] path 实验目录或检查点。
    @param[in] task 期望的 YOLO 任务。
    @return Ultralytics 模型。
    """
    model = YOLO(str(yolo_weights(path)))
    check_geometry(model.ckpt)
    if model.task != task:
        raise ValueError(f"Expected {task} checkpoint, got {model.task}")
    return model


def train_yolo(data, out, task="detect", size="n", epochs=20, imgsz=320, batch=8, device="cpu", seed=20261118):
    """@brief 从 Ultralytics COCO 发布权重开始训练，保留在验证集上选出的权重。

    @param[in] data 已完成的训练清单目录。
    @param[in] out 存放权重、标签与 experiment.yaml 的空实验目录。
    @param[in] task detect 或 pose。
    @param[in] size YOLO11 模型规模。
    @param[in] epochs 训练预算，即遍历数据集的轮数。
    @param[in] imgsz letterbox 后的网络输入边长，单位 pixel。
    @param[in] batch 每步图像数。
    @param[in] device Torch 设备名。
    @param[in] seed 可复现的训练种子。
    """
    data, out = Path(data).resolve(), Path(out).resolve()
    require_empty(out)
    dataset = yolo_dataset(data, out / "dataset", task)
    name = f"yolo11{size}" + ("-pose" if task == "pose" else "")
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}.pt"
    source = pretrained_file(out, url)
    model = YOLO(str(source))
    best_epoch = 0

    def record_epoch(trainer):
        """@brief 在 final_eval 去除优化器与轮次信息之前记录 Ultralytics 选中的轮次。"""
        nonlocal best_epoch
        if trainer.fitness == trainer.best_fitness:
            best_epoch = trainer.epoch + 1

    model.add_callback("on_model_save", record_epoch)
    model.train(
        data=str(dataset),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out.parent),
        name=out.name,
        exist_ok=True,
        seed=seed,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.05,
        nbs=batch,
        patience=epochs + 1,
        plots=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        close_mosaic=0,
        verbose=False,
    )
    report = experiment(data, task, seed, {**vars(model.trainer.args), "size": size})
    for filename in ("best.pt", "last.pt"):
        path = out / "weights" / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint["geometry_sha256"] = report["geometry_sha256"]
        torch.save(checkpoint, path)
    report.update(
        chosen_epoch=best_epoch,
        pretrained_url=url,
        pretrained_sha256=sha256(source),
        weights_sha256=sha256(yolo_weights(out)),
    )
    write_yaml(out / "experiment.yaml", report)


class YoloPosePredictor:
    """@brief 在 RGB 图像上直接做 YOLO11-pose 推理，取得分最高的检测框。"""

    def __init__(self, weights, device="cpu"):
        """@brief 在指定 Torch 设备上加载训练好的 pose 权重或运行目录。

        @param[in] weights pose 实验目录或检查点。
        @param[in] device Torch 设备。
        """
        ## 输出有序 16 点的直接 pose 模型。
        self.model = load_yolo(weights, "pose")
        ## Ultralytics 推理设备。
        self.device = device

    def predict(self, rgb):
        """@brief 仅用 RGB 推理原图检测框与有序 XY 关键点（单位 pixel）。

        @param[in] rgb 原始 RGB8 图像。
        @return 原图坐标下的预测，单位 pixel。
        """
        result = self.model.predict(
            np.ascontiguousarray(rgb[:, :, ::-1]),
            imgsz=self.model.ckpt["train_args"]["imgsz"],
            device=self.device,
            rect=False,
            conf=0.25,
            verbose=False,
        )[0]
        if not len(result.boxes):
            return Prediction.missing()
        i = int(result.boxes.conf.argmax())
        points = result.keypoints.data[i].cpu().numpy()
        return Prediction(result.boxes.xyxy[i].cpu().numpy(), float(result.boxes.conf[i]), points[:, :2], points[:, 2])
