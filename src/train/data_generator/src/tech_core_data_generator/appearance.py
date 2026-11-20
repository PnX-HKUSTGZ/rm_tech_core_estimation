"""@brief Blender Cycles 光照、色调映射与材质参数。"""

import numpy as np

from .camera import Viewpoint

## 参考 Cycles 影棚。位置为核心系坐标（单位 m），灯光功率单位 W，颜色为线性 RGB。
DEFAULT_APPEARANCE = {
    "view_transform": "Standard",
    "exposure": 0.0,
    "light_energy": 180.0,
    "light_position": (2.0, -1.0, 3.0),
    "world_strength": 0.4,
    "world_color": (0.025, 0.03, 0.04),
    "transparent_background": False,
    "material_scale": 1.0,
    "metallic": 0.25,
    "roughness": 0.4,
    "emission_scale": 1.0,
}


def sample_appearance(rng: np.random.Generator, view: Viewpoint) -> dict:
    """@brief 采样域随机化 Blender 数据集使用的训练外观。

    @param[in] rng 该划分使用的带种子随机数流。
    @param[in] view 相机视角，用于相机侧光照分量。
    @return 可序列化的外观覆盖参数，供 TechCoreSimulator.render() 使用。
    """
    if rng.random() < 0.55:
        centre = np.array([-0.1, 0, 0.2])
        direction = view.position - centre
        position = centre + 2.5 * direction / np.linalg.norm(direction) + rng.uniform(-0.3, 0.3, 3)
    else:
        position = rng.uniform([0.5, -2, 0.3], [2.5, 2, 3])
    ambient = rng.uniform(0.65, 1)
    return {
        "view_transform": "Raw" if rng.random() < 0.65 else "Standard",
        "exposure": rng.uniform(-0.5, 0.5),
        "light_energy": float(np.exp(rng.uniform(np.log(50), np.log(220)))),
        "light_position": position.tolist(),
        "world_strength": rng.uniform(0.25, 0.65),
        "world_color": [ambient] * 3,
        "transparent_background": True,
        "material_scale": rng.uniform(0.65, 1.4),
        "metallic": rng.uniform(0, 0.35),
        "roughness": rng.uniform(0.4, 1),
        "emission_scale": rng.uniform(0.6, 1.5),
    }
