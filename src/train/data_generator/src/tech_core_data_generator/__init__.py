"""@brief 渲染带标注的科技核心图像，并读取其几何与数据集契约。"""

from .camera import CameraConfig, Viewpoint
from .dataset import load_manifest
from .geometry import geometry_contract, geometry_sha256
from .lighting import LightState, Team
from .simulator import TechCoreSimulator

__all__ = [
    "CameraConfig",
    "LightState",
    "Team",
    "TechCoreSimulator",
    "Viewpoint",
    "geometry_contract",
    "geometry_sha256",
    "load_manifest",
]
