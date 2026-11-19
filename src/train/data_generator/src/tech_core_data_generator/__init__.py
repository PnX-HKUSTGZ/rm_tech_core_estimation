"""@brief Render labelled technology core images and read their geometry and dataset contracts."""

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
