"""@brief Pretrained YOLO11 detector and direct 16-point pose baseline."""

from pathlib import Path

import numpy as np
import torch
from tech_core_data_generator.dataset import require_empty
from ultralytics import YOLO

from .labels import check_geometry, experiment, pretrained_file, sha256, write_yaml, yolo_dataset
from .prediction import Prediction


def yolo_weights(path):
    """@brief Resolve a YOLO experiment directory or an explicit checkpoint.

    @param[in] path Detector or pose experiment directory or checkpoint.
    @return Checkpoint path.
    """
    path = Path(path)
    return path / "weights/best.pt" if path.is_dir() else path


def load_yolo(path, task):
    """@brief Load a trained checkpoint with the canonical landmark identity and requested task.

    @param[in] path Experiment directory or checkpoint.
    @param[in] task Expected YOLO task.
    @return Ultralytics model.
    """
    model = YOLO(str(yolo_weights(path)))
    check_geometry(model.ckpt)
    if model.task != task:
        raise ValueError(f"Expected {task} checkpoint, got {model.task}")
    return model


def train_yolo(data, out, task="detect", size="n", epochs=20, imgsz=320, batch=8, device="cpu", seed=20261118):
    """@brief Train from the Ultralytics COCO release, keeping validation-selected weights.

    @param[in] data Completed training manifest directory.
    @param[in] out Empty experiment directory for weights, labels and experiment.yaml.
    @param[in] task detect or pose.
    @param[in] size YOLO11 model scale.
    @param[in] epochs Training budget in passes over the dataset.
    @param[in] imgsz Letterboxed network input side (pixel).
    @param[in] batch Images per step.
    @param[in] device Torch device name.
    @param[in] seed Reproducible training seed.
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
        """@brief Capture Ultralytics' selected epoch before final_eval strips optimizer and epoch."""
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
    """@brief Direct YOLO11-pose inference on an RGB image using the highest-scoring box."""

    def __init__(self, weights, device="cpu"):
        """@brief Load the trained pose weights or run directory on the given Torch device.

        @param[in] weights Pose experiment or checkpoint.
        @param[in] device Torch device.
        """
        ## Direct pose model with ordered 16-point output.
        self.model = load_yolo(weights, "pose")
        ## Ultralytics inference device.
        self.device = device

    def predict(self, rgb):
        """@brief Infer original-image box and ordered XY landmarks (pixel) from RGB alone.

        @param[in] rgb Original RGB8 image.
        @return Prediction in original image coordinates (pixel).
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
