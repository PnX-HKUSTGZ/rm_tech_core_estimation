"""@brief Pinhole camera calibration, projection and source-view distributions."""

from dataclasses import dataclass
from math import cos, pi, sin, tan

import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """@brief Centred square-pixel camera with zero distortion.

    @param[in] width Image width (pixel).
    @param[in] height Image height (pixel).
    @param[in] fovy Vertical field of view (rad).
    @param[in] near Near optical clipping depth (m).
    @param[in] far Far optical clipping depth (m).
    """

    width: int = 320
    height: int = 240
    fovy: float = pi / 3
    near: float = 0.005
    far: float = 10.0

    def __post_init__(self):
        """@brief Reject camera settings outside the pinhole and clipping domain."""
        if min(self.width, self.height) < 1 or not 0 < self.fovy < pi or not 0 < self.near < self.far:
            raise ValueError("Require positive image size, 0 < fovy < pi rad and 0 < near < far")

    @property
    def matrix(self) -> np.ndarray:
        """@brief Compute pinhole intrinsics.

        @return 3×3 optical camera matrix (pixel), with integer pixel centres.
        """
        focal = self.height / (2 * tan(self.fovy / 2))
        return np.array([[focal, 0, (self.width - 1) / 2], [0, focal, (self.height - 1) / 2], [0, 0, 1]])


@dataclass(frozen=True)
class Viewpoint:
    """@brief Camera looking towards a point in the nominal core frame.

    @param[in] r Distance from the core origin (m).
    @param[in] theta Azimuth about +Z from +X towards +Y (rad).
    @param[in] alpha Elevation above the core XY plane (rad).
    @param[in] target Look-at point in the core frame (m).
    @param[in] roll Rotation around the optical axis (rad).
    """

    r: float = 0.8
    theta: float = 0.0
    alpha: float = 0.0
    target: tuple[float, float, float] = (-0.112, 0.0, 0.2046324834)
    roll: float = 0.0

    def __post_init__(self):
        """@brief Reject invalid camera positions before passing them to a renderer."""
        if self.r <= 0 or not np.isfinite([self.r, self.theta, self.alpha, self.roll, *self.target]).all():
            raise ValueError("Viewpoint requires a positive radius and finite coordinates and angles")

    @property
    def position(self) -> np.ndarray:
        """@brief Convert spherical coordinates to Cartesian coordinates.

        @return Camera origin in the core frame (m).
        """
        return self.r * np.array(
            [cos(self.alpha) * cos(self.theta), cos(self.alpha) * sin(self.theta), sin(self.alpha)]
        )

    @property
    def camera_from_core(self) -> np.ndarray:
        """@brief Construct a rolled look-at camera.

        @return Core pose in the optical frame (4×4, translation in m, +Z forward and +Y down).
        @throws ValueError When the look-at point coincides with the camera.
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
    """@brief Project landmarks into the undistorted original image.

    @param[in] points Core-frame XYZ coordinates (m).
    @param[in] camera Image calibration and optical clipping limits.
    @param[in] camera_from_core Core pose in the optical frame (translation in m).
    @return Pixel coordinates (NaN behind camera), optical depths (m) and in-frame flags.
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
    """@brief Draw one independent source view using the data_generator Readme distributions.

    @param[in] rng Seeded random stream for this split.
    @param[in] distribution regular, stress or training (85% common views).
    @return Sampled core-frame viewpoint (m, rad).
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
