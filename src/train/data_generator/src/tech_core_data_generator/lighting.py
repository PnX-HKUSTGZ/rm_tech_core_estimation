"""@brief Absolute-time lamp states from RoboMaster 2027 rule table 5-13."""

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from math import floor

import numpy as np

from .geometry import MODEL_DIR, read_obj


class Team(StrEnum):
    """@brief Team colour of the main and pillar lamps."""

    RED = "red"  ##< Red team.
    BLUE = "blue"  ##< Blue team.


class LightState(StrEnum):
    """@brief Assembly step indicated by the lamps, with the marker in its initial pose."""

    IDLE = "idle"  ##< First group off.
    MOVING = "moving"  ##< Running white bars.
    READY = "ready"  ##< All groups blink at 1 Hz.
    STEP2_COMPLETE = "step2_complete"  ##< Pillar group blinks at 3 Hz.
    STEP3_COMPLETE = "step3_complete"  ##< Main and pillar groups blink at 3 Hz.
    STEP4_COMPLETE = "step4_complete"  ##< Main and pillar groups steady.
    STEP5_ROTATING = "step5_rotating"  ##< Target and current angles indicated.
    STEP5_COMPLETE = "step5_complete"  ##< Target angle indicated in green.
    RECOVERING = "recovering"  ##< All groups blink at 3 Hz.


GROUP1 = tuple(f"core_back_light_bar_{i}" for i in range(1, 37))  ##< Angle display bars.
GROUP2 = tuple(f"core_back_main_light_{i}" for i in range(1, 3))  ##< Main lamps.
GROUP3 = tuple(f"core_pillar_L_light_{i}" for i in range(1, 7)) + tuple(
    f"core_pillar_side_light_{i}" for i in range(1, 4)
)  ##< Pillar lamps.


@dataclass(frozen=True)
class LightConfig:
    """@brief Simulation assumptions for colours and timings unspecified by the rule."""

    duty_cycle: float = 0.5  ##< Powered fraction of a blink period.
    chase_hz: float = 1.0  ##< Full sweeps per second.
    chase_width: int = 3  ##< Number of active consecutive bars.
    red: tuple = (1.0, 0.005, 0.012)  ##< Red team linear RGB.
    blue: tuple = (0.005, 0.08, 1.0)  ##< Blue team linear RGB.
    white: tuple = (1.0, 1.0, 1.0)  ##< First group's linear RGB.
    green: tuple = (0.005, 1.0, 0.02)  ##< Successful angle indication, linear RGB.
    off: tuple = (0.015, 0.015, 0.015)  ##< Unpowered diffuse linear RGB.


@dataclass(frozen=True)
class Lamp:
    """@brief Material assigned to one lamp mesh."""

    rgb: tuple  ##< Linear RGB.
    on: bool  ##< Whether this lamp emits light.


@cache
def _bar_angles() -> np.ndarray:
    """@brief Measure bar centres around the assembly rotation axis in the core frame (rad)."""
    centres = []
    for index in range(1, 37):
        vertices, _ = read_obj(MODEL_DIR / "meshes" / f"back_light_bar_{index}.obj")
        centres.append(vertices.mean(axis=0))
    centres = np.asarray(centres)
    return np.arctan2(-centres[:, 1], centres[:, 2] - 0.208)


class LightController:
    """@brief Evaluate rule-defined light groups at arbitrary timestamps."""

    def __init__(self, config: LightConfig = LightConfig()):
        """@brief Set the rule-table simulation parameters.

        @param[in] config Timing and material assumptions.
        """
        self.config = config

    def angle_bar(self, angle: float) -> str:
        """@brief Quantize the displayed angle to its nearest physical bar.

        @param[in] angle Right-handed rotation about core +X (rad).
        @return Geometry name, using the lower index for a tie.
        """
        if not -np.pi / 2 <= angle <= np.pi / 2:
            raise ValueError("Indicated rotation must be within [-pi/2, pi/2] rad")
        distances = np.abs(_bar_angles() - angle)
        return GROUP1[np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-9))[0]]

    def evaluate(self, state: LightState, team: Team, time_s: float, target_angle: float = 0, current_angle: float = 0):
        """@brief Evaluate the 47 lamps without advancing a simulation clock.

        @param[in] state Assembly light state.
        @param[in] team Indicated team colour.
        @param[in] time_s Absolute lighting time (s).
        @param[in] target_angle Desired rotation about core +X (rad).
        @param[in] current_angle Displayed current rotation about core +X (rad).
        @return Lamp materials indexed by MJCF geometry name.
        """
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Lighting time must be finite and nonnegative (s)")
        config = self.config
        color = config.red if team == Team.RED else config.blue
        off = Lamp(config.off, False)
        lamps = dict.fromkeys(GROUP1, off) | dict.fromkeys(GROUP2 + GROUP3, Lamp(color, True))

        def blink(group, rgb, frequency):
            """@brief Evaluate the square wave at its exact half/full-period boundaries."""
            phase = round(time_s * frequency, 12) % 1
            lamps.update(dict.fromkeys(group, Lamp(rgb, True) if phase < config.duty_cycle else off))

        if state == LightState.MOVING:
            start = floor(round(time_s * config.chase_hz * len(GROUP1), 12))
            for index in range(start, start + config.chase_width):
                lamps[GROUP1[index % len(GROUP1)]] = Lamp(config.white, True)
        elif state in (
            LightState.READY,
            LightState.STEP2_COMPLETE,
            LightState.STEP3_COMPLETE,
            LightState.STEP4_COMPLETE,
        ):
            blink(GROUP1, config.white, 1)
            if state != LightState.STEP4_COMPLETE:
                blink(GROUP2, color, 3 if state == LightState.STEP3_COMPLETE else 1)
                blink(GROUP3, color, 1 if state == LightState.READY else 3)
        elif state == LightState.STEP5_ROTATING:
            lamps[self.angle_bar(current_angle)] = Lamp(config.white, True)
            lamps[self.angle_bar(target_angle)] = Lamp(color, True)
        elif state == LightState.STEP5_COMPLETE:
            lamps[self.angle_bar(target_angle)] = Lamp(config.green, True)
        elif state == LightState.RECOVERING:
            blink(GROUP1, config.white, 3)
            blink(GROUP2 + GROUP3, color, 3)
        return lamps
