"""@brief YOLO dataset conversion and training provenance for the YAML dataset contract."""

import hashlib
import platform
import shutil
import urllib.request
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml
from tech_core_data_generator.dataset import load_manifest
from tech_core_data_generator.geometry import geometry_sha256


def sha256(path: Path) -> str:
    """@brief Return the SHA-256 of a manifest, checkpoint or experiment file.

    @param[in] path Input file.
    @return Lowercase hex digest.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_yaml(path, data):
    """@brief Publish a completed YAML artifact atomically in its destination directory.

    @param[in] path Destination file.
    @param[in] data YAML-compatible content.
    """
    with TemporaryDirectory(dir=path.parent) as work:
        temporary = Path(work) / path.name
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.dump(data, stream, Dumper=yaml.CSafeDumper, sort_keys=False)
        temporary.replace(path)


def read_yaml(path):
    """@brief Read an experiment, selection or evaluation artifact.

    @param[in] path Source YAML file.
    @return Parsed YAML document.
    """
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def check_geometry(checkpoint):
    """@brief Reject weights or selections with a missing or different ordered geometry fingerprint.

    @param[in] checkpoint Metadata containing geometry_sha256.
    """
    if checkpoint.get("geometry_sha256") != geometry_sha256():
        raise ValueError("Checkpoint or selection geometry SHA-256 differs from the installed geometry contract")


def pretrained_file(out, url):
    """@brief Cache upstream weights next to the experiment and return the local path.

    @param[in] out Training experiment directory.
    @param[in] url Upstream release asset URL.
    @return Local checkpoint path.
    """
    directory = out.parent / "pretrained"
    directory.mkdir(exist_ok=True)
    path = directory / url.rsplit("/", 1)[1]
    if not path.exists():
        with TemporaryDirectory(dir=directory) as work:
            temporary = Path(work) / path.name
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
    return path


def experiment(data, task, seed, hyperparameters):
    """@brief Record versions, seed, training parameters and identities of training inputs.

    @param[in] data Training manifest directory.
    @param[in] task Training command.
    @param[in] seed Training seed.
    @param[in] hyperparameters Optimizer and model configuration.
    @return Experiment metadata document.
    """
    packages = (
        "torch",
        "torchvision",
        "ultralytics",
        "mmengine",
        "mmcv-lite",
        "numpy",
        "tech-core-model-train",
        "tech-core-data-generator",
        "tech-core-data-enhancement",
        "onnx",
        "onnxruntime",
    )
    return {
        "task": task,
        "geometry_sha256": geometry_sha256(),
        "manifest_sha256": sha256(data / "manifest.yaml"),
        "seed": seed,
        "hyperparameters": hyperparameters,
        "versions": {
            **{p: version(p) for p in packages},
            "python": platform.python_version(),
            "opencv": cv2.__version__,
        },
    }


def rgb_image(path):
    """@brief Load an original uint8 RGB image from a manifest path.

    @param[in] path Source image file.
    @return H×W×3 RGB8 image.
    """
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def image_points(row):
    """@brief Read 16 original image points (pixel), replacing manifest null coordinates with NaN.

    @param[in] row Positive manifest record.
    @return 16×2 original-image coordinates (pixel).
    """
    return np.array([p if p is not None else [np.nan, np.nan] for p in row["pixels"]], dtype=float)


def yolo_label(row, task):
    """@brief Encode a manifest observation as a normalized detect or 16-point pose label.

    @param[in] row Manifest record with original-image annotations (pixel).
    @param[in] task detect or pose.
    @return YOLO label line, or empty string for background.
    """
    if row["bbox_xyxy"] is None:
        return ""
    box = np.asarray(row["bbox_xyxy"])
    size = np.array([row["width"], row["height"]])
    values = [0, *((box[:2] + box[2:]) / (2 * size)), *((box[2:] - box[:2]) / size)]
    if task == "pose":
        for point, flag in zip(image_points(row), row["visibility"], strict=True):
            values.extend((*point / size, flag) if flag and (point >= 0).all() and (point < size).all() else (0, 0, 0))
    return " ".join(f"{v:.8f}" for v in values) + "\n"


def yolo_dataset(data: Path, out: Path, task: str) -> Path:
    """@brief Write split image/label directories and return the Ultralytics dataset YAML path.

    @param[in] data Completed Blender dataset root.
    @param[in] out Destination for derived images and labels.
    @param[in] task detect includes backgrounds, pose includes target boxes only.
    @return Path to the YOLO dataset configuration.
    """
    splits = {}
    for row in load_manifest(data)["records"]:
        split = row["split"]
        if split not in ("train", "val", "test") or (task == "pose" and (row["negative"] or row["bbox_xyxy"] is None)):
            continue
        image = Path(row["image"])
        if split not in splits:
            splits[split] = f"images/{split}"
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
        shutil.copyfile(data / image, out / "images" / split / (row["id"] + image.suffix))
        (out / "labels" / split / (row["id"] + ".txt")).write_text(yolo_label(row, task))
    config = {"path": str(out.resolve()), **splits, "names": {0: "tech_core"}}
    if task == "pose":
        config["kpt_shape"] = [16, 3]
    path = out / f"{task}.yaml"
    write_yaml(path, config)
    return path
