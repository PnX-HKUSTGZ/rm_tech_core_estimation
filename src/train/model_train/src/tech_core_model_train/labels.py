"""@brief 面向 YAML 数据集契约的 YOLO 数据集转换与训练来源记录。"""

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
    """@brief 返回清单、检查点或实验文件的 SHA-256。

    @param[in] path 输入文件。
    @return 小写十六进制摘要。
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_yaml(path, data):
    """@brief 在目标目录中原子地发布一个完整的 YAML 产物。

    @param[in] path 目标文件。
    @param[in] data 可序列化为 YAML 的内容。
    """
    with TemporaryDirectory(dir=path.parent) as work:
        temporary = Path(work) / path.name
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.dump(data, stream, Dumper=yaml.CSafeDumper, sort_keys=False)
        temporary.replace(path)


def read_yaml(path):
    """@brief 读取实验、选择或评估产物。

    @param[in] path 源 YAML 文件。
    @return 解析后的 YAML 文档。
    """
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.CSafeLoader)


def check_geometry(checkpoint):
    """@brief 拒绝缺少有序几何指纹或指纹不一致的权重与选择结果。

    @param[in] checkpoint 含 geometry_sha256 的元数据。
    """
    if checkpoint.get("geometry_sha256") != geometry_sha256():
        raise ValueError("Checkpoint or selection geometry SHA-256 differs from the installed geometry contract")


def pretrained_file(out, url):
    """@brief 将上游权重缓存到实验目录旁，并返回本地路径。

    @param[in] out 训练实验目录。
    @param[in] url 上游发布资源 URL。
    @return 本地检查点路径。
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
    """@brief 记录版本、种子、训练参数及训练输入的标识。

    @param[in] data 训练清单目录。
    @param[in] task 训练子命令。
    @param[in] seed 训练种子。
    @param[in] hyperparameters 优化器与模型配置。
    @return 实验元数据文档。
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
    """@brief 由清单路径加载原始 uint8 RGB 图像。

    @param[in] path 源图像文件。
    @return H×W×3 RGB8 图像。
    """
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def image_points(row):
    """@brief 读取 16 个原图关键点（单位 pixel），清单中的 null 坐标替换为 NaN。

    @param[in] row 正样本清单记录。
    @return 16×2 原图坐标，单位 pixel。
    """
    return np.array([p if p is not None else [np.nan, np.nan] for p in row["pixels"]], dtype=float)


def yolo_label(row, task):
    """@brief 把清单观测编码为归一化的 detect 标签或 16 点 pose 标签。

    @param[in] row 带原图标注（单位 pixel）的清单记录。
    @param[in] task detect 或 pose。
    @return YOLO 标签行；背景样本返回空字符串。
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
    """@brief 写出各划分的图像/标签目录，并返回 Ultralytics 数据集 YAML 路径。

    @param[in] data 已完成的 Blender 数据集根目录。
    @param[in] out 派生图像与标签的输出目录。
    @param[in] task detect 包含背景样本，pose 只包含有目标框的样本。
    @return YOLO 数据集配置文件路径。
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
