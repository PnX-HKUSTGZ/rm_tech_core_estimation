"""@brief 源视角采样与已完成数据集的 YAML 读写。"""

from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .appearance import sample_appearance
from .camera import CameraConfig, sample_random
from .geometry import _digest, geometry_contract, geometry_sha256
from .lighting import LightState, Team
from .simulator import TechCoreSimulator


def load_manifest(directory: str | Path) -> dict:
    """@brief 按 data_generator Readme 的规定加载已完成的数据集。

    @param[in] directory 含 manifest.yaml 的数据集根目录。
    @return 路径相对于数据集根目录的清单。
    @throws FileNotFoundError 数据集没有最终清单时抛出。
    @throws ValueError schema 或几何不兼容，或不同划分共用源视角时抛出。
    """
    with (Path(directory) / "manifest.yaml").open(encoding="utf-8") as file:
        manifest = yaml.load(file, Loader=yaml.CSafeLoader)
    if manifest["schema_version"] != 1:
        raise ValueError(f"Unsupported dataset schema version {manifest['schema_version']}")
    geometry = manifest["geometry"]
    if geometry["sha256"] != geometry_sha256() or _digest(geometry) != geometry["sha256"]:
        raise ValueError("Dataset geometry or SHA-256 differs from the installed geometry contract")
    owners = {}
    for record in manifest["records"]:
        view, split = record["source_view"], record["split"]
        if owners.setdefault(view, split) != split:
            raise ValueError(f"Source view {view!r} appears in multiple splits")
    return manifest


def require_empty(out: Path) -> None:
    """@brief 创建输出目录，或接受已存在的空目录。

    @param[in] out 请求的输出路径。
    @throws FileExistsError 目录中已有任何条目时抛出。
    """
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")


def _pose_key(camera_from_core) -> tuple:
    """@brief 按参考留出集的精度比较已存储与新采样的相机位姿。"""
    return tuple(np.round(camera_from_core, 12).ravel())


def generate_dataset(
    out: Path,
    renderer: str,
    counts: dict[str, int],
    seed: int,
    camera: CameraConfig,
    argv: list[str],
    *,
    samples: int = 8,
    appearance: str | None = None,
    exclude=(),
):
    """@brief 渲染相互独立的源视角，并原子地发布最终清单。

    @param[in] out 新建或为空的数据集根目录。
    @param[in] renderer blender 用于训练数据，mujoco 用于留出集。
    @param[in] counts 各划分（train/val/test 或 regular/stress）的源视角数量。
    @param[in] seed 独立的非负运行种子。
    @param[in] camera 原图标定（单位 pixel、rad、m）。
    @param[in] argv 记录在生成器来源信息中的命令行参数。
    @param[in] samples Blender Cycles 每像素采样数。
    @param[in] appearance Blender 外观策略（random 或 default）。
    @param[in] exclude 需要排除其位姿的已完成数据集。
    """
    if min(counts.values()) < 0 or not any(counts.values()):
        raise ValueError("Split counts must be nonnegative and request at least one view")
    randomize = renderer == "blender" and appearance != "default"
    known = set()
    for directory in exclude:
        manifest = load_manifest(directory)
        known.update(_pose_key(row["camera_from_core"]) for row in manifest["records"] if not row["negative"])
    records = []
    states = list(LightState)
    offsets = {"train": 0, "val": 100003, "test": 200003, "regular": 0, "stress": 100003}
    require_empty(out)
    with (
        (out / "records.partial.yaml").open("x", encoding="utf-8") as journal,
        TechCoreSimulator(renderer, camera, blender_samples=samples) as simulator,
    ):
        (out / "images").mkdir()
        (out / "masks").mkdir()
        generator = {
            "version": version("tech-core-data-generator"),
            "argv": argv,
            "seed": seed,
            "blender_version": simulator.blender_version,
        }
        for split, count in counts.items():
            distribution = "training" if renderer == "blender" else split
            split_seed = seed + offsets[split]
            rng = np.random.default_rng(split_seed)
            # 比较不同外观策略时，固定源相机与灯光时刻。
            appearance_rng = np.random.default_rng([split_seed, 1])
            for index in range(count):
                while True:
                    view = sample_random(rng, distribution)
                    pose = _pose_key(view.camera_from_core)
                    if pose not in known:
                        known.add(pose)
                        break
                settings = sample_appearance(appearance_rng, view) if randomize else None
                frame = simulator.render(
                    view,
                    states[index % len(states)],
                    Team.RED if index % 2 else Team.BLUE,
                    time_s=rng.uniform(0, 3),
                    appearance=settings,
                )
                identity = f"{renderer}_{seed}_{split}_{index:05d}"
                record = frame.to_record(identity, split, identity)
                Image.fromarray(frame.rgb).save(out / record["image"])
                Image.fromarray(frame.mask.astype(np.uint8) * 255).save(out / record["mask"])
                yaml.dump(record, journal, Dumper=yaml.CSafeDumper, explicit_start=True, sort_keys=False)
                journal.flush()
                records.append(record)
    manifest = {
        "schema_version": 1,
        "geometry": {**geometry_contract(), "sha256": geometry_sha256()},
        "generator": generator,
        "records": records,
    }
    temporary = out / "manifest.partial.yaml"
    with temporary.open("x", encoding="utf-8") as file:
        yaml.dump(manifest, file, Dumper=yaml.CSafeDumper, sort_keys=False)
    temporary.replace(out / "manifest.yaml")
