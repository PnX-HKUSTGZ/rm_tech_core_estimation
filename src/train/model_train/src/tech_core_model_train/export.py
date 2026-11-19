"""@brief Export the fixed two-stage deployment graphs and replay held-out golden frames."""

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

CONFIDENCE_TOLERANCE = 1e-3  ##< Allowed PyTorch/ONNX per-point confidence difference.


def export_graphs(detector, keypoints, directory):
    """@brief Export eval-mode CPU modules with fixed RGB inputs and raw candidate, heatmap and logit outputs.

    @param[in] detector CPU YOLO module producing raw candidates.
    @param[in] keypoints CPU LiteHRNet module producing heatmaps and logits.
    @param[in] directory Destination for the two fixed-shape ONNX files.
    """
    for model, side, filename, names in (
        (detector, 320, "detector", ["candidates"]),
        (keypoints, 256, "keypoints", ["heatmaps", "logits"]),
    ):
        path = directory / f"{filename}.onnx"
        tensor = torch.zeros(1, 3, side, side)
        with torch.no_grad():
            model(tensor)  # Materialize the detector's anchors for the fixed input size.
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
    """@brief Run the exported graphs on golden RGBs and compare flags, boxes, confidence and landmark XY.

    @param[in] directory Release containing both ONNX files, model_meta.yaml and golden/cases.yaml.
    @return Hit count and maximum end-to-end landmark deviation (pixel).
    @throws ValueError For any detection disagreement, landmark deviation of at least 0.5 pixel,
            or golden frames without a single detection.
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
        # float32 graph differences reach ~1e-4 in confidence; a validity flip within that band of the threshold is noise.
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
    """@brief Sample up to 16 evaluation frames across splits and hits/misses; store CPU PyTorch predictions.

    The evaluation may have run on CUDA; the reference is recomputed on CPU, the deployment target of the check.
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
    """@brief Publish a YOLO11n + LiteHRNet-18 release from evaluated, frozen checkpoints.

    @param[in] detector Trained detector run or weights/best.pt.
    @param[in] keypoints Trained LiteHRNet run or best.pt.
    @param[in] evaluation Matching completed evaluation directory, with access to its source images.
    @param[in] out Empty release directory.
    @param[in] verify Replay ONNX Runtime before publishing.
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
