"""@brief 共用的渲染器门面及基于几何的真值计算。"""

from copy import deepcopy

import numpy as np

from .camera import CameraConfig, Viewpoint, project_points
from .frame import Frame
from .geometry import geometry_contract, unobstructed
from .lighting import LightController, LightState, Team


class TechCoreSimulator:
    """@brief 用一个可复用的后端与光学相机渲染静止的核心。"""

    def __init__(self, backend: str = "mujoco", camera: CameraConfig | None = None, blender_samples: int = 8):
        """@brief 打开离屏渲染器。

        @param[in] backend mujoco 或 blender。
        @param[in] camera 针孔标定（单位 pixel、rad、m）。
        @param[in] blender_samples Cycles 每像素采样数。
        """
        self.backend = backend
        self.camera = camera or CameraConfig()
        self._points = np.array(geometry_contract()["points_m"])
        self._lights = LightController()
        self.blender_version = None
        if backend == "mujoco":
            from .backends.mujoco_backend import MujocoBackend

            self._renderer = MujocoBackend(self.camera)
        elif backend == "blender":
            from .backends.blender_backend import BlenderBackend

            if blender_samples < 1:
                raise ValueError("Cycles samples must be positive")
            self._renderer = BlenderBackend(self.camera, blender_samples)
            self.blender_version = self._renderer.version
        else:
            raise ValueError(f"Unknown renderer {backend!r}")

    def render(
        self,
        view: Viewpoint | None = None,
        state: LightState | str = LightState.IDLE,
        team: Team | str = Team.RED,
        time_s: float = 0.0,
        target_angle: float = 0.0,
        current_angle: float = 0.0,
        appearance: dict | None = None,
    ) -> Frame:
        """@brief 渲染一个视角，并计算投影关键点与基于三角形的遮挡。

        @param[in] view 核心系中的光学相机视角（单位 m、rad）。
        @param[in] state 规则表灯光状态。
        @param[in] team 指示的队伍颜色。
        @param[in] time_s 灯光绝对时刻，单位 s。
        @param[in] target_angle 绕核心系 +X 的目标旋转角，单位 rad。
        @param[in] current_angle 显示的绕核心系 +X 的旋转角，单位 rad。
        @param[in] appearance Blender 材质、灯光与颜色覆盖参数；为 None 时使用默认值。
        @return 含独立图像与标注数组的帧。
        @throws ValueError MuJoCo 后端收到外观覆盖参数时抛出。
        """
        state, team = LightState(state), Team(team)
        lamps = self._lights.evaluate(state, team, time_s, target_angle, current_angle)
        view = view or Viewpoint()
        camera_from_core = view.camera_from_core
        if self.backend == "blender":
            rgb, depth = self._renderer.render(camera_from_core, lamps, appearance)
        else:
            if appearance is not None:
                raise ValueError("Appearance overrides require the Blender backend")
            rgb, depth = self._renderer.render(camera_from_core, lamps)
        pixels, _, inside = project_points(self._points, self.camera, camera_from_core)
        visibility = inside.astype(np.uint8)
        visibility[inside] += unobstructed(view.position, self._points[inside])
        return Frame(
            rgb,
            depth,
            self.camera,
            camera_from_core,
            pixels,
            visibility,
            self.backend,
            state,
            team,
            time_s,
            deepcopy(appearance),
        )

    def close(self):
        """@brief 释放后端资源。"""
        self._renderer.close()

    def __enter__(self):
        """@brief 进入受管理的渲染会话。

        @return 本模拟器。
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """@brief 会话结束时关闭渲染器。

        @param[in] exc_type 当前异常类型（如有）。
        @param[in] exc_value 异常实例（如有）。
        @param[in] traceback 当前调用栈（如有）。
        """
        self.close()
