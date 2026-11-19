"""@brief Shared renderer facade and geometry-based ground truth."""

from copy import deepcopy

import numpy as np

from .camera import CameraConfig, Viewpoint, project_points
from .frame import Frame
from .geometry import geometry_contract, unobstructed
from .lighting import LightController, LightState, Team


class TechCoreSimulator:
    """@brief Render a stationary core with one reusable backend and optical camera."""

    def __init__(self, backend: str = "mujoco", camera: CameraConfig | None = None, blender_samples: int = 8):
        """@brief Open an offscreen renderer.

        @param[in] backend mujoco or blender.
        @param[in] camera Pinhole calibration (pixel, rad, m).
        @param[in] blender_samples Cycles samples per pixel.
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
        """@brief Render one view with projected landmarks and triangle-based occlusion.

        @param[in] view Optical camera viewpoint in the core frame (m, rad).
        @param[in] state Rule-table lamp state.
        @param[in] team Indicated team colour.
        @param[in] time_s Absolute lighting time (s).
        @param[in] target_angle Desired rotation around core +X (rad).
        @param[in] current_angle Displayed rotation around core +X (rad).
        @param[in] appearance Blender material, light and colour overrides, or None for defaults.
        @return Frame containing independent image and label arrays.
        @throws ValueError If MuJoCo receives an appearance override.
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
        """@brief Release backend resources."""
        self._renderer.close()

    def __enter__(self):
        """@brief Enter a managed rendering session.

        @return This simulator.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """@brief Close the renderer when the session ends.

        @param[in] exc_type Active exception type, if any.
        @param[in] exc_value Exception instance, if any.
        @param[in] traceback Active traceback, if any.
        """
        self.close()
