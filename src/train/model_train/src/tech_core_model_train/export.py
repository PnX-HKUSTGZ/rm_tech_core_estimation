"""@brief 导出固定的两阶段部署计算图，并回放留出的金标准帧。"""

import shutil
from collections import defaultdict
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tech_core_data_generator.dataset import load_manifest, require_empty
from tech_core_data_generator.geometry import geometry_contract, geometry_sha256

from .evaluate import model_hashes, score_prediction
from .labels import check_geometry, read_yaml, rgb_image, sha256, write_yaml
from .litehrnet.predict import DetectorLitePredictor, best_box, crop_input, decode_prediction, letterbox, lite_weights
from .pnp import KeypointModel
from .yolo import yolo_weights

CONFIDENCE_TOLERANCE = 1e-3  ##< PyTorch 与 ONNX 逐点置信度允许的差异。


def export_graphs(detector, keypoints, directory):
    """@brief 导出 eval 模式的 CPU 模块，输入为固定尺寸 RGB，输出原始候选框、热图与 logit。

    @param[in] detector 输出原始候选框的 CPU YOLO 模块。
    @param[in] keypoints 输出热图与 logit 的 CPU LiteHRNet 模块。
    @param[in] directory 两个固定形状 ONNX 文件的输出目录。
    """
    for model, side, filename, names in (
        (detector, 320, "detector", ["candidates"]),
        (keypoints, 256, "keypoints", ["heatmaps", "logits"]),
    ):
        path = directory / f"{filename}.onnx"
        tensor = torch.zeros(1, 3, side, side)
        with torch.no_grad():
            model(tensor)  # 按固定输入尺寸生成检测器的锚点。
            torch.onnx.export(
                model,
                tensor,
                path,
                opset_version=17,
                dynamo=False,
                input_names=["rgb"],
                output_names=names,
            )
        graph = onnx.load(path)
        onnx.helper.set_model_props(graph, {"geometry_sha256": geometry_sha256()})
        onnx.save(graph, path)


def verify_release(directory):
    """@brief 在金标准 RGB 上运行导出的计算图，比对检出标记、检测框、置信度与关键点 XY。

    @param[in] directory 含两个 ONNX 文件、model_meta.yaml 与 golden/cases.yaml 的发布目录。
    @return 检出数量及端到端关键点最大偏差，单位 pixel。
    @throws ValueError 检测结果不一致、关键点偏差达到 0.5 pixel，
            或金标准帧中没有任何检出时抛出。
    """
    directory = Path(directory)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    detector, keypoints = [
        ort.InferenceSession(str(directory / f"{name}.onnx"), options, providers=["CPUExecutionProvider"])
        for name in ("detector", "keypoints")
    ]
    meta = read_yaml(directory / "model_meta.yaml")
    if meta["geometry"] != {**geometry_contract(), "sha256": geometry_sha256()}:
        raise ValueError("Release geometry differs from the installed geometry contract")
    for session in (detector, keypoints):
        check_geometry(session.get_modelmeta().custom_metadata_map)
    errors = []
    cases = read_yaml(directory / "golden/cases.yaml")
    for case in cases:
        rgb = rgb_image(directory / "golden" / case["image"])
        h, w = rgb.shape[:2]
        image, gain, pad = letterbox(rgb, meta["detector"]["input_size"])
        (candidates,) = detector.run(None, {"rgb": image})
        box, score = best_box(candidates, gain, pad, w, h)
        if (box is None) != (case["bbox_xyxy"] is None):
            raise ValueError(f"Detection differs for {case['image']}")
        if box is None:
            continue
        if np.max(abs(box - case["bbox_xyxy"])) >= 0.5 or abs(score - case["score"]) > 1e-4:
            raise ValueError(f"Detection box or score differs for {case['image']}")
        crop, transform = crop_input(rgb, box)
        maps, logits = keypoints.run(None, {"rgb": crop})
        pred = decode_prediction(box, score, maps, logits, transform)
        deviation = float(np.linalg.norm(pred.pixels - case["pixels"], axis=1).max())
        if not deviation < 0.5:
            raise ValueError(f"Keypoints differ by {deviation:g} pixel for {case['image']}")
        threshold = meta["keypoints"]["threshold"]
        # float32 计算图的置信度差异可达约 1e-4；门限附近该范围内的有效性翻转视为噪声。
        decisive = np.abs(np.asarray(case["confidence"]) - threshold) > CONFIDENCE_TOLERANCE
        flips = pred.valid(w, h, threshold) != np.asarray(case["valid"])
        if np.any(flips & decisive) or not np.allclose(
            pred.confidence, case["confidence"], atol=CONFIDENCE_TOLERANCE, rtol=0
        ):
            raise ValueError(f"Keypoint confidence or validity differs for {case['image']}")
        errors.append(deviation)
    if not errors:
        raise ValueError("Golden frames contain no detection; the release cannot be verified")
    return {"detections": len(errors), "end_to_end_max_px": max(errors)}


def _golden(directory, data, manifest, frames, predictor, threshold):
    """@brief 跨划分、兼顾检出/漏检抽取至多 16 个评估帧，保存 CPU 上 PyTorch 的预测。

    评估可能在 CUDA 上运行；参考结果在 CPU 上重算，CPU 是该校验面向的部署目标。
    """
    groups = defaultdict(list)
    for frame in frames:
        groups[(frame["renderer"], frame["split"], frame["bbox_xyxy"] is not None)].append(frame)
    sampled = [frame for batch in zip_longest(*groups.values()) for frame in batch if frame is not None][:16]
    rows = {row["id"]: row for row in manifest["records"]}
    model = KeypointModel()
    directory.mkdir()
    cases = []
    for frame in sampled:
        row = rows[frame["id"]]
        image = Path(row["image"])
        name = row["id"] + image.suffix
        shutil.copyfile(data / image, directory / name)
        frame = score_prediction(row, predictor.predict(rgb_image(data / image)), threshold, model)
        fields = ("id", "bbox_xyxy", "score", "pixels", "confidence", "valid", "camera_from_core")
        cases.append(
            {k: frame[k] for k in fields}
            | {"image": name, "camera_matrix": row["camera_matrix"], "distortion": row["distortion"]}
        )
    write_yaml(directory / "cases.yaml", cases)


def export(detector, keypoints, evaluation, out, verify=False):
    """@brief 由已评估、已冻结的检查点发布 YOLO11n + LiteHRNet-18 发布目录。

    @param[in] detector 训练好的检测器运行目录或 weights/best.pt。
    @param[in] keypoints 训练好的 LiteHRNet 运行目录或 best.pt。
    @param[in] evaluation 对应的已完成评估目录，需能访问其源图像。
    @param[in] out 空的发布目录。
    @param[in] verify 发布前用 ONNX Runtime 回放校验。
    """
    evaluation, out = Path(evaluation), Path(out)
    weights = model_hashes(detector, keypoints)
    summary = read_yaml(evaluation / "summary.yaml")
    selection = read_yaml(evaluation / "selection.yaml")
    check_geometry(selection)
    if weights != selection["weights_sha256"]:
        raise ValueError("Export weights SHA-256 differs from the evaluated selection")
    threshold = selection["keypoint_threshold"]
    data = Path(summary["data"])
    manifest = load_manifest(data)
    experiment_files = {
        "detector": yolo_weights(detector).parent.parent / "experiment.yaml",
        "keypoints": lite_weights(keypoints).parent / "experiment.yaml",
    }
    experiments = {name: read_yaml(path) for name, path in experiment_files.items()}
    predictor = DetectorLitePredictor(detector, keypoints, "cpu")
    network = predictor.network
    if predictor.imgsz != 320 or experiments["detector"]["hyperparameters"]["size"] != "n" or network.variant != 18:
        raise ValueError("Deployment requires YOLO11n at 320 pixel and LiteHRNet-18")
    meta = {
        "schema_version": 1,
        "geometry": manifest["geometry"],
        "detector": {
            "input_size": 320,
            "channel_order": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "normalization": "divide_by_255",
            "threshold": 0.25,
            "letterbox": {
                "gain": "320/max(height,width)",
                "resized_size": "round_to_nearest_even",
                "interpolation": "opencv_linear",
                "pad_value": 114,
                "left_top": "floor(remaining/2)",
                "center": True,
            },
            "output": "[1,5,2100] center_x,center_y,width,height,score; letterbox pixels",
        },
        "keypoints": {
            "crop_size": 256,
            "padding": 1.25,
            "heatmap_size": 128,
            "channel_order": "RGB",
            "input_range": [0, 255],
            "mean": network.mean.flatten().tolist(),
            "std": network.std.flatten().tolist(),
            "normalization_in_graph": True,
            "threshold": threshold,
            "confidence": "sigmoid(logit)*clip(peak,0,1)",
        },
        "provenance": {
            "experiment_sha256": {name: sha256(path) for name, path in experiment_files.items()},
            "evaluation": summary["domains"],
            "exported_at_utc": datetime.now(UTC).isoformat(),
        },
    }
    require_empty(out)
    with TemporaryDirectory(dir=out.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        export_graphs(predictor.detector, network, staging)
        write_yaml(staging / "model_meta.yaml", meta)
        _golden(staging / "golden", data, manifest, read_yaml(evaluation / "frames.yaml"), predictor, threshold)
        if verify:
            print("Golden verification", verify_release(staging), flush=True)
        staging.replace(out)
