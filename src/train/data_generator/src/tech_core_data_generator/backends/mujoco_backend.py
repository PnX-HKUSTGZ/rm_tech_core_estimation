"""@brief 固定装配体的 MuJoCo 离屏渲染。"""

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..camera import CameraConfig
from ..geometry import MODEL_DIR
from ..lighting import GROUP1, GROUP2, GROUP3


class MujocoBackend:
    """@brief 为单个相机持有模型、数据与离屏 OpenGL 上下文。"""

    def __init__(self, camera: CameraConfig):
        """@brief 将模型冻结在初始关节位置，并添加光学相机。

        @param[in] camera 光学标定（单位 pixel）与裁剪距离（单位 m）。
        """
        root = ET.parse(MODEL_DIR / "tech_core.xml").getroot()
        for mesh in root.findall("asset/mesh"):
            mesh.set("file", str(MODEL_DIR / mesh.get("file")))
        for parent in root.iter():
            for child in list(parent):
                if child.tag in ("joint", "inertial") or (child.tag == "geom" and child.get("group") == "3"):
                    parent.remove(child)
        visual = root.find("visual")
        visual.find("global").attrib.update(offwidth=str(camera.width), offheight=str(camera.height))
        extent = float(root.find("statistic").get("extent"))
        ET.SubElement(visual, "map", znear=str(camera.near / extent), zfar=str(camera.far / extent))
        ET.SubElement(
            root.find("worldbody"), "camera", name="dataset_camera", fovy=str(math.degrees(camera.fovy)), ipd="0"
        )
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.camera_id = self.model.camera("dataset_camera").id
        self.materials = {name: self.model.geom(name).matid.item() for name in GROUP1 + GROUP2 + GROUP3}
        self.renderer = mujoco.Renderer(self.model, height=camera.height, width=camera.width)
        self.far = camera.far

    def render(self, camera_from_core, lamps):
        """@brief 按核心位姿与灯光映射渲染 RGB8 与光学深度（单位 m，背景为零）。

        @param[in] camera_from_core 核心系在相机光学系中的位姿，平移单位 m。
        @param[in] lamps 按网格名索引的材质。
        @return 独立持有的 RGB 与光学深度数组。
        """
        core_from_camera = camera_from_core[:3, :3].T
        self.model.cam_pos[self.camera_id] = -core_from_camera @ camera_from_core[:3, 3]
        core_from_render_camera = core_from_camera @ np.diag([1.0, -1.0, -1.0])
        mujoco.mju_mat2Quat(self.model.cam_quat[self.camera_id], core_from_render_camera.ravel())
        for name, lamp in lamps.items():
            material = self.materials[name]
            self.model.mat_rgba[material] = (*lamp.rgb, 1)
            self.model.mat_emission[material] = float(lamp.on)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera_id)
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        depth[depth >= self.far * 0.9999] = 0
        return rgb, depth

    def close(self):
        """@brief 释放 OpenGL 上下文。"""
        self.renderer.close()
