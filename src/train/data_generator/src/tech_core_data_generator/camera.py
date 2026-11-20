"""@brief 针孔相机标定、投影与源视角分布。"""

from dataclasses import dataclass
from math import cos, pi, sin, tan

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """@brief 主点居中、方形像素、无畸变的相机。

    @param[in] width 图像宽度，单位 pixel。
    @param[in] height 图像高度，单位 pixel。
    @param[in] fovy 垂直视场角，单位 rad。
    @param[in] near 光学近裁剪深度，单位 m。
    @param[in] far 光学远裁剪深度，单位 m。
    """

    width: int = 320
    height: int = 240
    fovy: float = pi / 3
    near: float = 0.005
    far: float = 10.0

    def __post_init__(self):
        """@brief 拒绝超出针孔模型与裁剪范围的相机参数。"""
        if min(self.width, self.height) < 1 or not 0 < self.fovy < pi or not 0 < self.near < self.far:
            raise ValueError("Require positive image size, 0 < fovy < pi rad and 0 < near < far")

    @property
    def matrix(self) -> np.ndarray:
        """@brief 计算针孔内参。

        @return 3×3 相机矩阵，单位 pixel，像素中心位于整数坐标。
        """
        focal = self.height / (2 * tan(self.fovy / 2))
        return np.array([[focal, 0, (self.width - 1) / 2], [0, focal, (self.height - 1) / 2], [0, 0, 1]])


@dataclass(frozen=True)
class Viewpoint:
    """@brief 朝向名义核心系中某一点的相机。

    @param[in] r 到核心系原点的距离，单位 m。
    @param[in] theta 绕 +Z 的方位角，由 +X 转向 +Y，单位 rad。
    @param[in] alpha 相对核心系 XY 平面的仰角，单位 rad。
    @param[in] target 核心系中的注视点，单位 m。
    @param[in] roll 绕光轴的滚转角，单位 rad。
    """

    r: float = 0.8
    theta: float = 0.0
    alpha: float = 0.0
    target: tuple[float, float, float] = (-0.112, 0.0, 0.2046324834)
    roll: float = 0.0

    def __post_init__(self):
        """@brief 在交给渲染器之前拒绝非法的相机位置。"""
        if self.r <= 0 or not np.isfinite([self.r, self.theta, self.alpha, self.roll, *self.target]).all():
            raise ValueError("Viewpoint requires a positive radius and finite coordinates and angles")

    @property
    def position(self) -> np.ndarray:
        """@brief 将球坐标转换为直角坐标。

        @return 相机原点在核心系中的位置，单位 m。
        """
        return self.r * np.array(
            [cos(self.alpha) * cos(self.theta), cos(self.alpha) * sin(self.theta), sin(self.alpha)]
        )

    @property
    def camera_from_core(self) -> np.ndarray:
        """@brief 构造带滚转的注视相机。

        @return 核心系在相机光学系中的位姿（4×4，平移单位 m，+Z 向前、+Y 向下）。
        @throws ValueError 注视点与相机位置重合时抛出。
        """
        forward = self.target - self.position
        length = np.linalg.norm(forward)
        if length == 0:
            raise ValueError("Camera position and look-at target must differ")
        forward /= length
        up = [0, 0, 1] if abs(forward[2]) < 1 - 1e-8 else [0, 1, 0]
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        cosine, sine = cos(self.roll), sin(self.roll)
        rotation = np.stack([cosine * right + sine * down, -sine * right + cosine * down, forward])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = -rotation @ self.position
        return transform


def project_points(points: np.ndarray, camera: CameraConfig, camera_from_core: np.ndarray):
    """@brief 将关键点投影到无畸变原图。

    @param[in] points 核心系 XYZ 坐标，单位 m。
    @param[in] camera 图像标定与光学裁剪范围。
    @param[in] camera_from_core 核心系在相机光学系中的位姿，平移单位 m。
    @return 像素坐标（相机后方为 NaN）、光学深度（单位 m）及是否在图像内的标记。
    """
    optical = points @ camera_from_core[:3, :3].T + camera_from_core[:3, 3]
    depth = optical[:, 2]
    front = depth > 0
    pixels = np.full((len(points), 2), np.nan)
    pixels[front] = (optical[front] @ camera.matrix.T)[:, :2] / depth[front, None]
    inside = (depth >= camera.near) & (depth <= camera.far)
    inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
    return pixels, depth, inside


def sample_random(rng: np.random.Generator, distribution: str) -> Viewpoint:
    """@brief 按 data_generator Readme 中的分布采样一个独立源视角。

    @param[in] rng 该划分使用的带种子随机数流。
    @param[in] distribution regular、stress 或 training（85% 为常见视角）。
    @return 采样得到的核心系视角（单位 m、rad）。
    """
    if distribution == "training":
        if rng.random() < 0.15:
            radius, azimuth, elevation = 0.25, 1.40, (-0.96, 0.96)
        else:
            radius, azimuth, elevation = 0.4, 1.13, (-0.52, 0.79)
        roll, lateral, height = pi / 12, 0.03, (0.17, 0.24)
    else:
        radius, azimuth, elevation = {
            "regular": (0.45, 0.96, (-0.35, 0.61)),
            "stress": (0.2, 1.57, (-1.05, 1.05)),
        }[distribution]
        roll, lateral, height = pi / 18, 0.025, (0.18, 0.23)
    return Viewpoint(
        r=rng.uniform(radius, 1),
        theta=rng.uniform(-azimuth, azimuth),
        alpha=rng.uniform(*elevation),
        roll=rng.uniform(-roll, roll),
        target=(-0.112, rng.uniform(-lateral, lateral), rng.uniform(*height)),
    )
