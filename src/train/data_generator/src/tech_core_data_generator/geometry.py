"""@brief 核心系关键点及用于自遮挡计算的可视网格。"""

import hashlib
import json
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent / "models" / "tech_core"  ##< 随包分发的 MJCF 与核心系网格。


@cache
def _definition() -> dict:
    """@brief 加载随包分发的关键点文件；测试将外壳点固定到图 5-9 的作图结果。"""
    # Blender 在其自带 Python 中导入网格读取模块，该环境没有 PyYAML。
    import yaml

    with Path(__file__).with_name("keypoints.yaml").open() as file:
        data = yaml.load(file, Loader=yaml.CSafeLoader)
    return {
        "profile": data["profile"],
        "names": [point["name"] for point in data["points"]],
        "groups": [point["group"] for point in data["points"]],
        "points_m": [point["xyz"] for point in data["points"]],
    }


def geometry_contract() -> dict:
    """@brief 返回有序 16 关键点定义的独立副本。

    @return 配置名、名称、分组及对齐的名义核心系 XYZ 坐标（单位 m）。
    """
    return deepcopy(_definition())


def _digest(contract: dict) -> str:
    """@brief 按 geometry_sha256() 中描述的规范形式编码几何数据。"""
    payload = {key: contract[key] for key in ("profile", "names", "groups")}
    payload["points_m"] = [[round(value, 9) for value in point] for point in contract["points_m"]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@cache
def geometry_sha256() -> str:
    """@brief 计算几何指纹，供数据集与模型元数据使用。

    @return 包含 profile、names、groups 与 points_m 的 UTF-8 JSON 的小写 SHA-256。
    每个坐标（单位 m）四舍五入到小数点后 9 位，保持数组顺序，对象键排序，
    使用 Python json.dumps 且 separators 为 (',', ':')，末尾不加换行。
    """
    return _digest(_definition())


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """@brief 读取随包分发的三角化 OBJ 文件，不做坐标轴或单位转换。

    @param[in] path 可视网格路径。
    @return 核心系顶点（单位 m）及从零开始的三角形顶点下标。
    """
    vertices, faces = [], []
    with path.open() as mesh:
        for line in mesh:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(corner.split("/")[0]) - 1 for corner in line.split()[1:]])
    return np.array(vertices), np.array(faces, dtype=np.int32)


@cache
def _triangles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@brief 收集零关节位核心系下的可视三角形顶点与边（单位 m）。"""
    parts = []
    for path in sorted((MODEL_DIR / "meshes").glob("*.obj")):
        vertices, faces = read_obj(path)
        parts.append(vertices[faces])
    triangles = np.concatenate(parts)
    return triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]


def unobstructed(camera_position: np.ndarray, points: np.ndarray) -> np.ndarray:
    """@brief 用 Möller–Trumbore 三角形求交检测相机到关键点的线段。

    @param[in] camera_position 光心在核心系中的位置，单位 m。
    @param[in] points 图像内关键点在核心系中的位置，单位 m。
    @return 每个关键点一个布尔值；距关键点 0.001 m 以内的交点视为其自身表面。
    """
    anchor, edge1, edge2 = _triangles()
    offset = camera_position - anchor
    cross = np.cross(offset, edge1)
    numerator = np.einsum("ij,ij->i", edge2, cross)
    visible = []
    for point in points:
        ray = point - camera_position
        length = np.linalg.norm(ray)
        direction = ray / length
        normal = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, normal)
        intersects = np.abs(determinant) > 1e-12
        inverse = np.divide(1, determinant, where=intersects, out=np.zeros_like(determinant))
        u = np.einsum("ij,ij->i", offset, normal) * inverse
        v = np.einsum("ij,j->i", cross, direction) * inverse
        distance = numerator * inverse
        intersects &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        intersects &= (distance > 1e-7) & (distance < length - 0.001)
        visible.append(not np.any(intersects))
    return np.array(visible, dtype=bool)
