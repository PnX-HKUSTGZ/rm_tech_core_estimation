"""@brief LiteHRNet training on Blender crops and development-set PnP model selection."""

from math import cos, pi
from pathlib import Path

import numpy as np
import torch
from tech_core_data_enhancement import photometric
from tech_core_data_generator.dataset import load_manifest, require_empty
from torch.utils.data import DataLoader, Dataset

from ..evaluate import calibrate, calibration_rows, model_hashes, predictions
from ..labels import experiment, image_points, pretrained_file, rgb_image, sha256, write_yaml
from ..pnp import KeypointModel
from ..prediction import square_crop
from ..yolo import yolo_weights
from .model import LiteKeypoints, decode_heatmaps, heatmap_targets
from .predict import DetectorLitePredictor, lite_weights, load_keypoints


class CropDataset(Dataset):
    """@brief Positive Blender crops with aligned heatmap centres and in-frame labels."""

    def __init__(self, root, rows, split, seed, augment=False):
        """@brief Select the requested split and seed its training-only jitter and photometry.

        @param[in] root Blender dataset directory.
        @param[in] rows Manifest records.
        @param[in] split train for jitter or val for fixed crops.
        @param[in] seed Seed for crop jitter and photometry.
        @param[in] augment Enable online photometry.
        """
        ## Directory for manifest-relative image paths.
        self.root = root
        ## Positive source records in this split.
        self.rows = [
            r
            for r in rows
            if r["renderer"] == "blender" and r["split"] == split and not r["negative"] and r["bbox_xyxy"] is not None
        ]
        ## Enable geometric crop jitter.
        self.training = split == "train"
        ## Enable online appearance augmentation.
        self.augment = augment
        ## Seeded augmentation generator.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """@brief Return the number of positive crops in this split.

        @return Number of positive crops.
        """
        return len(self.rows)

    def __getitem__(self, index):
        """@brief Return float RGB CHW, heatmaps, in-frame flags and crop coordinates (pixel).

        @param[in] index Index in this split.
        @return Crop, heatmaps, in-frame mask and crop-space targets (pixel).
        """
        row = self.rows[index]
        box = np.array(row["bbox_xyxy"])
        if self.training:
            size = box[2:] - box[:2]
            center = (box[2:] + box[:2]) / 2 + self.rng.uniform(-0.05, 0.05, 2) * size
            size = size * self.rng.uniform(0.9, 1.1)
            box = np.r_[center - size / 2, center + size / 2]
        crop, transform = square_crop(rgb_image(self.root / row["image"]), box)
        if self.augment:
            crop = photometric(crop, self.rng)
        points = image_points(row) @ transform[:2, :2].T + transform[:2, 2]
        valid = (np.asarray(row["visibility"]) > 0) & (points >= 0).all(axis=1) & (points < 256).all(axis=1)
        return crop.transpose(2, 0, 1).astype(np.float32), heatmap_targets(points, valid), valid, points


@torch.inference_mode()
def _crop_error(loader, network, device):
    """@brief Mean Euclidean error of in-frame Blender validation landmarks in crop pixels."""
    errors = []
    for rgb, _, valid, truth in loader:
        maps, _ = network(rgb.to(device))
        for heatmaps, mask, points in zip(maps.cpu().numpy(), valid.numpy(), truth.numpy(), strict=True):
            xy, _ = decode_heatmaps(heatmaps)
            errors.extend(np.linalg.norm(xy[mask] - points[mask], axis=1))
    if not errors:
        raise ValueError("Crop checkpoint selection requires in-frame landmarks in Blender val")
    return float(np.mean(errors))


def _fit(data, out, variant, epochs, batch, learning_rate, seed, device, initial=None, dev_data=None, detector=None):
    """@brief Fit heatmaps and visibility logits, saving the epoch selected on the designated split."""
    data, out = Path(data).resolve(), Path(out).resolve()
    manifest = load_manifest(data)
    if epochs < 1 or batch < 2 or learning_rate <= 0:
        raise ValueError("LiteHRNet training requires epochs >= 1, batch >= 2 and positive learning rate")
    dataset = CropDataset(data, manifest["records"], "train", seed, augment=initial is not None)
    if len(dataset) < 2:
        raise ValueError("LiteHRNet batch normalization requires at least two positive Blender training images")
    require_empty(out)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    params = {
        "epochs": epochs,
        "batch": batch,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": 0.05,
        "visibility_loss_weight": 0.0001,
        "gradient_norm_limit": 10.0,
        "cosine_lr_floor": 0.05,
        "device": device,
    }
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        drop_last=len(dataset) % batch == 1,
        generator=torch.Generator().manual_seed(seed),
    )
    if initial:
        network = load_keypoints(initial)
        dev_data = Path(dev_data).resolve()
        dev_rows = calibration_rows(load_manifest(dev_data), True)
        predictor = DetectorLitePredictor(detector, network, device)
        geometry = KeypointModel()
        report = experiment(data, "finetune", seed, params)
        report.update(
            initial_sha256=sha256(lite_weights(initial)),
            detector_sha256=sha256(yolo_weights(detector)),
            development_manifest_sha256=sha256(dev_data / "manifest.yaml"),
        )
    else:
        network = LiteKeypoints(variant)
        names = {
            18: "litehrnet18_coco_256x192-6bace359_20211230.pth",
            30: "litehrnet30_coco_256x192-4176555b_20210626.pth",
        }
        url = "https://download.openmmlab.com/mmpose/top_down/litehrnet/" + names[variant]
        source = pretrained_file(out, url)
        network.load_pretrained_backbone(source)
        validation = DataLoader(CropDataset(data, manifest["records"], "val", seed), batch_size=batch)
        report = experiment(data, "keypoints", seed, params)
        report.update(pretrained_url=url, pretrained_sha256=sha256(source))
    params["variant"] = network.variant
    network.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in network.parameters() if p.ndim > 1], "weight_decay": 0.05},
            {"params": [p for p in network.parameters() if p.ndim <= 1], "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    best, history = None, []
    for epoch in range(epochs):
        lr = learning_rate * (0.05 + 0.95 * (1 + cos(pi * epoch / epochs)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        network.train()
        total, count = 0.0, 0
        for rgb, targets, valid, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            maps, logits = network(rgb.to(device))
            heatmap_loss = (maps - targets.to(device)).square().mean()
            visibility_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, valid.to(device, dtype=torch.float32)
            )
            loss = heatmap_loss + 0.0001 * visibility_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
            optimizer.step()
            total += loss.item() * len(rgb)
            count += len(rgb)
        network.eval()
        entry = {"epoch": epoch + 1, "learning_rate": lr, "loss": total / count}
        if initial:
            accurate, pck, threshold = calibrate(list(predictions(dev_data, dev_rows, predictor)), geometry)
            key = (accurate, pck)
            entry.update(dev_accurate=accurate, dev_pck5=pck, keypoint_threshold=threshold)
        else:
            error = _crop_error(validation, network, device)
            key = (-error,)
            entry["val_crop_error_px"] = error
        checkpoint = {
            "state_dict": network.state_dict(),
            "geometry_sha256": report["geometry_sha256"],
            "variant": network.variant,
            "epoch": epoch + 1,
        }
        torch.save(checkpoint, out / "last.pt")
        if best is None or key > best:
            best = key
            report["chosen_epoch"] = epoch + 1
            torch.save(checkpoint, out / "best.pt")
            if initial:
                write_yaml(
                    out / "selection.yaml",
                    {
                        "geometry_sha256": report["geometry_sha256"],
                        "source": "mujoco_dev",
                        "calibration_sha256": report["development_manifest_sha256"],
                        "chosen_epoch": epoch + 1,
                        "weights_sha256": model_hashes(detector, out),
                        "keypoint_threshold": threshold,
                    },
                )
        history.append(entry)
        print(entry, flush=True)
    report.update(history=history, weights_sha256=sha256(out / "best.pt"))
    write_yaml(out / "experiment.yaml", report)


def train(data, out, variant=18, epochs=40, batch=16, learning_rate=0.001, seed=20261118, device="cpu"):
    """@brief Transfer the COCO backbone and select weights by Blender validation crop error (pixel).

    @param[in] data Completed Blender training dataset.
    @param[in] out Empty run directory for weights and provenance.
    @param[in] variant LiteHRNet depth, 18 or 30.
    @param[in] epochs Training passes over the dataset.
    @param[in] batch Maximum images per optimizer step.
    @param[in] learning_rate Peak AdamW learning rate.
    @param[in] seed Seed for crop jitter, initialization and shuffling.
    @param[in] device Torch device.
    """
    _fit(data, out, variant, epochs, batch, learning_rate, seed, device)


def finetune(
    data, out, initial, dev_data, detector, epochs=30, batch=16, learning_rate=0.0001, seed=20261118, device="cpu"
):
    """@brief Fine-tune on photometrically augmented Blender crops with epoch/threshold selection by development PnP.

    @param[in] data Completed Blender training dataset.
    @param[in] out Empty run directory for weights, provenance and frozen selection.
    @param[in] initial Canonical LiteHRNet checkpoint or experiment.
    @param[in] dev_data MuJoCo dataset reserved for model selection.
    @param[in] detector Frozen detector checkpoint or experiment.
    @param[in] epochs Training passes over the dataset.
    @param[in] batch Maximum images per optimizer step.
    @param[in] learning_rate Peak AdamW learning rate.
    @param[in] seed Seed for training augmentation and shuffling.
    @param[in] device Torch device.
    """
    _fit(data, out, None, epochs, batch, learning_rate, seed, device, initial, dev_data, detector)
