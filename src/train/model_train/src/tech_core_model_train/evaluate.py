"""@brief 在对留出 RGB 帧打分之前冻结验证集/开发集上的选择。"""

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from tech_core_data_generator.dataset import load_manifest, require_empty
from tech_core_data_generator.geometry import geometry_sha256

from .labels import check_geometry, image_points, read_yaml, rgb_image, sha256, write_yaml
from .litehrnet.predict import DetectorLitePredictor, lite_weights
from .pnp import KeypointModel, estimate
from .yolo import YoloPosePredictor, yolo_weights


def score_prediction(row, prediction, threshold, model, recognition_s=0.0):
    """@brief 用带标注的原图为一次纯 RGB 预测打分。

    @param[in] row 清单记录，含标定与真值。
    @param[in] prediction 检测框与有序的原图关键点，单位 pixel。
    @param[in] threshold 关键点置信度下限。
    @param[in] model 有序的核心系几何，单位 m。
    @param[in] recognition_s RGB 推理墙钟耗时，单位 s。
    @return 帧结果，含有效性、参考位姿、误差与失败类别。
    """
    box = prediction.bbox_xyxy
    valid = prediction.valid(row["width"], row["height"], threshold)
    start = perf_counter()
    pose = (
        estimate(model, prediction.pixels, valid, row["camera_matrix"], row["distortion"]) if box is not None else None
    )
    elapsed = recognition_s + perf_counter() - start
    iou = 0.0
    if box is not None and row["bbox_xyxy"] is not None:
        truth = np.array(row["bbox_xyxy"])
        overlap = np.maximum(0, np.minimum(box[2:], truth[2:]) - np.maximum(box[:2], truth[:2])).prod()
        iou = float(overlap / ((box[2:] - box[:2]).prod() + (truth[2:] - truth[:2]).prod() - overlap))
    translation = rotation = failure = None
    visible_count = correct_points = 0
    if not row["negative"]:
        visible = np.asarray(row["visibility"]) == 2
        visible_count = int(visible.sum())
        errors = np.linalg.norm(prediction.pixels - image_points(row), axis=1)
        correct_points = int((valid & visible & (errors <= 5)).sum())
        if pose is None:
            failure = "no_box" if box is None else "box_unsolved"
        else:
            actual = np.asarray(row["camera_from_core"])
            translation = float(np.linalg.norm(pose[:3, 3] - actual[:3, 3]))
            rotation = float(np.arccos(np.clip((np.trace(pose[:3, :3] @ actual[:3, :3].T) - 1) / 2, -1, 1)))
            if translation > 0.01 or rotation > 0.0873:
                failure = "solved_out_of_tolerance"
    return {
        "id": row["id"],
        "image": row["image"],
        "split": row["split"],
        "renderer": row["renderer"],
        "negative": row["negative"],
        "bbox_xyxy": None if box is None else box.tolist(),
        "score": prediction.score,
        "pixels": [p.tolist() if np.isfinite(p).all() else None for p in prediction.pixels],
        "confidence": prediction.confidence.tolist(),
        "valid": valid.tolist(),
        "iou": iou,
        "visible_points": visible_count,
        "pck5_count": correct_points,
        "camera_from_core": None if pose is None else pose.tolist(),
        "translation_m": translation,
        "rotation_rad": rotation,
        "failure": failure,
        "elapsed_s": elapsed,
    }


def aggregate(frames):
    """@brief 汇总检测、关键点与位姿指标；位姿准确率的分母包含全部目标帧。

    @param[in] frames 来自同一渲染器与划分的已打分记录。
    @return 计数、比率、Wilson 区间及条件误差分位数（单位 m、rad、s）。
    """
    positives = [f for f in frames if not f["negative"]]
    n = len(positives)
    negatives = len(frames) - n
    tp = sum(f["iou"] >= 0.5 for f in positives)
    detected = sum(f["bbox_xyxy"] is not None for f in frames)
    background_fp = sum(f["negative"] and f["bbox_xyxy"] is not None for f in frames)
    points = sum(f["visible_points"] for f in positives)
    solved = sum(f["camera_from_core"] is not None for f in positives)
    accurate = sum(f["failure"] is None for f in positives)
    interval = None
    if n:
        p, z = accurate / n, 1.959963984540054
        center = p + z * z / (2 * n)
        radius = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
        interval = [max(0.0, (center - radius) / (1 + z * z / n)), min(1.0, (center + radius) / (1 + z * z / n))]
    summary = {
        "positive_frames": n,
        "negative_frames": negatives,
        "detection_tp": tp,
        "detection_fp": detected - tp,
        "detection_recall": tp / n if n else None,
        "detection_precision": tp / detected if detected else None,
        "background_false_positives": background_fp,
        "negative_false_positive_rate": background_fp / negatives if negatives else None,
        "pck5": sum(f["pck5_count"] for f in positives) / points if points else None,
        "solved": solved,
        "solve_rate": solved / n if n else None,
        "accurate": accurate,
        "accurate_rate": accurate / n if n else None,
        "accuracy_wilson95": interval,
        "failures": {
            k: sum(f["failure"] == k for f in positives) for k in ("no_box", "box_unsolved", "solved_out_of_tolerance")
        },
    }
    for field in ("translation_m", "rotation_rad", "elapsed_s"):
        values = [f[field] for f in frames if f[field] is not None]
        for stat, q in [("median", 0.5), ("p95", 0.95)]:
            summary[f"{field}_{stat}"] = float(np.quantile(values, q)) if values else None
    return summary


def predictions(data, rows, predictor):
    """@brief 逐个产出记录、RGB 预测及推理墙钟耗时（单位 s，不含磁盘读写）。

    @param[in] data 清单目录。
    @param[in] rows 按顺序处理的记录。
    @param[in] predictor 纯 RGB 模型。
    @return (记录, 预测, 推理耗时 s) 的迭代器。
    """
    for row in rows:
        rgb = rgb_image(data / row["image"])
        start = perf_counter()
        pred = predictor.predict(rgb)
        yield row, pred, perf_counter() - start


def calibration_rows(manifest, development):
    """@brief 选出用于门限选择的 Blender val 帧，或显式指定的 MuJoCo 开发集帧。

    @param[in] manifest 已校验的数据集。
    @param[in] development 为 True 时使用显式保留的 MuJoCo 开发集。
    @return 符合条件的记录。
    """
    allowed = {("mujoco", "regular"), ("mujoco", "stress")} if development else {("blender", "val")}
    rows = [r for r in manifest["records"] if (r["renderer"], r["split"]) in allowed]
    if not rows:
        raise ValueError("Threshold calibration requires Blender val or an explicit MuJoCo --dev-data set")
    return rows


def calibrate(samples, model):
    """@brief 按位姿准确数选择置信度门限，其次比较可见点 PCK，再取较高门限。

    @param[in] samples 在标定划分上由 predictions() 缓存的观测。
    @param[in] model 有序的核心系模型，单位 m。
    @return 胜出的准确数、PCK 与门限。
    """
    choices = []
    for threshold in (0.05, 0.1, 0.25, 0.5):
        scores = [score_prediction(row, pred, threshold, model, elapsed) for row, pred, elapsed in samples]
        metrics = aggregate(scores)
        choices.append((metrics["accurate"], metrics["pck5"] or 0.0, threshold))
    return max(choices)


def model_hashes(detector, keypoints, pose=None):
    """@brief 确定与一次选择或评估绑定的推理检查点。

    @param[in] detector 两阶段方案的检测器运行目录或检查点。
    @param[in] keypoints 两阶段方案的关键点运行目录或检查点。
    @param[in] pose 直接回归方案的运行目录或检查点（若选用）。
    @return 角色到 SHA-256 的映射。
    """
    if pose:
        return {"pose": sha256(yolo_weights(pose))}
    return {"detector": sha256(yolo_weights(detector)), "keypoints": sha256(lite_weights(keypoints))}


def evaluate(
    data,
    out,
    detector=None,
    keypoints=None,
    pose=None,
    keypoint_threshold=None,
    selection=None,
    dev_data=None,
    device="cpu",
):
    """@brief 冻结模型与门限后，按渲染器与划分分别独立评估留出集。

    @param[in] data 含 Blender test 或 MuJoCo regular/stress 的清单根目录。
    @param[in] out 空输出目录，写入 selection.yaml、frames.yaml 与 summary.yaml。
    @param[in] detector 两阶段推理使用的 YOLO 检测器运行目录或检查点。
    @param[in] keypoints 与 detector 配对的 LiteHRNet 运行目录或检查点。
    @param[in] pose 基线方案使用的 YOLO pose 运行目录或检查点。
    @param[in] keypoint_threshold 预先声明的固定门限，取值 (0,1]。
    @param[in] selection 之前已冻结且绑定到这些检查点的 selection.yaml。
    @param[in] dev_data 可选标定使用的独立 MuJoCo 开发集清单。
    @param[in] device Torch 推理设备。
    @return 按渲染器与划分汇总的指标。
    """
    data, out = Path(data).resolve(), Path(out)
    manifest = load_manifest(data)
    digest = sha256(data / "manifest.yaml")
    weights = model_hashes(detector, keypoints, pose)
    predictor = YoloPosePredictor(pose, device) if pose else DetectorLitePredictor(detector, keypoints, device)
    model = KeypointModel()
    require_empty(out)
    if selection is None and keypoints:
        candidate = lite_weights(keypoints).parent / "selection.yaml"
        if candidate.exists():
            selection = candidate
    if selection:
        frozen = read_yaml(Path(selection))
        check_geometry(frozen)
        if frozen["weights_sha256"] != weights:
            raise ValueError("Selection weights SHA-256 differs from evaluation weights")
        if dev_data or (keypoint_threshold is not None and keypoint_threshold != frozen["keypoint_threshold"]):
            raise ValueError("Evaluation parameters disagree with the frozen selection")
    else:
        frozen = {"geometry_sha256": geometry_sha256(), "weights_sha256": weights, "source": "fixed"}
        if keypoint_threshold is None:
            calibration = Path(dev_data).resolve() if dev_data else data
            rows = calibration_rows(load_manifest(calibration) if dev_data else manifest, bool(dev_data))
            if dev_data and sha256(calibration / "manifest.yaml") == digest:
                raise ValueError("Development data cannot also be the evaluation set")
            _, _, keypoint_threshold = calibrate(list(predictions(calibration, rows, predictor)), model)
            frozen.update(
                source="mujoco_dev" if dev_data else "blender_val",
                calibration_sha256=sha256(calibration / "manifest.yaml"),
            )
        frozen["keypoint_threshold"] = keypoint_threshold
    if frozen["source"] == "mujoco_dev" and frozen["calibration_sha256"] == digest:
        raise ValueError("Development data cannot also be the evaluation set")
    threshold = frozen["keypoint_threshold"]
    if not 0 < threshold <= 1:
        raise ValueError("Keypoint threshold must be in (0,1]")
    write_yaml(out / "selection.yaml", frozen)
    rows = [r for r in manifest["records"] if r["split"] in ("test", "regular", "stress")]
    if not rows:
        raise ValueError("Evaluation requires held-out test, regular or stress records")
    frames = [
        score_prediction(row, pred, threshold, model, elapsed)
        for row, pred, elapsed in predictions(data, rows, predictor)
    ]
    domains = defaultdict(lambda: defaultdict(list))
    for frame in frames:
        domains[frame["renderer"]][frame["split"]].append(frame)
    summary = {
        "data": str(data),
        "manifest_sha256": digest,
        "weights_sha256": weights,
        "keypoint_threshold": threshold,
        "domains": {renderer: {s: aggregate(fs) for s, fs in splits.items()} for renderer, splits in domains.items()},
    }
    write_yaml(out / "frames.yaml", frames)
    write_yaml(out / "summary.yaml", summary)
    return summary
