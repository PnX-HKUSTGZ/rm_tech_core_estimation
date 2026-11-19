"""@brief Core-frame landmarks and the visual mesh used for self-occlusion."""

import hashlib
import json
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).parent / "models" / "tech_core"  ##< Packaged MJCF and core-frame meshes.


@cache
def _definition() -> dict:
    """@brief Load the packaged landmark file; tests pin the shell points to the Figure 5-9 construction."""
    # Blender imports the mesh reader in its bundled Python, which has no PyYAML.
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
    """@brief Return an independent copy of the ordered 16-landmark definition.

    @return Profile, names, groups and aligned XYZ coordinates in the nominal core frame (m).
    """
    return deepcopy(_definition())


def _digest(contract: dict) -> str:
    """@brief Encode the geometry payload using the canonical form documented by geometry_sha256()."""
    payload = {key: contract[key] for key in ("profile", "names", "groups")}
    payload["points_m"] = [[round(value, 9) for value in point] for point in contract["points_m"]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@cache
def geometry_sha256() -> str:
    """@brief Fingerprint the geometry for datasets and model metadata.

    @return Lowercase SHA-256 of UTF-8 JSON containing profile, names, groups and points_m.
    Round each coordinate to nine decimal places (m), retain array order, sort object keys,
    and use Python json.dumps with separators (',', ':') and no trailing newline.
    """
    return _digest(_definition())


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """@brief Read the packaged triangulated OBJ files without axis or unit conversion.

    @param[in] path Visual mesh path.
    @return Core-frame vertices (m) and zero-based triangle vertex indices.
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
    """@brief Collect the visual triangle anchors and edges in the zero-joint core frame (m)."""
    parts = []
    for path in sorted((MODEL_DIR / "meshes").glob("*.obj")):
        vertices, faces = read_obj(path)
        parts.append(vertices[faces])
    triangles = np.concatenate(parts)
    return triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]


def unobstructed(camera_position: np.ndarray, points: np.ndarray) -> np.ndarray:
    """@brief Test camera-to-landmark segments using Möller–Trumbore triangle intersection.

    @param[in] camera_position Optical centre in the core frame (m).
    @param[in] points In-frame landmark positions in the core frame (m).
    @return One boolean per landmark. Hits within 0.001 m of the landmark count as its own surface.
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
