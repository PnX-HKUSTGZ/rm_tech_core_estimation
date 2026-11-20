"""@brief 按 RoboMaster 2027 规则表 5-13 计算绝对时刻的灯光状态。"""

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from math import floor

import numpy as np

from .geometry import MODEL_DIR, read_obj


class Team(StrEnum):
    """@brief 主灯与立柱灯的队伍颜色。"""

    RED = "red"  ##< 红方。
    BLUE = "blue"  ##< 蓝方。


class LightState(StrEnum):
    """@brief 灯光指示的装配步骤，标识处于初始位姿。"""

    IDLE = "idle"  ##< 第一组熄灭。
    MOVING = "moving"  ##< 白色灯条流水。
    READY = "ready"  ##< 所有分组以 1 Hz 闪烁。
    STEP2_COMPLETE = "step2_complete"  ##< 立柱组以 3 Hz 闪烁。
    STEP3_COMPLETE = "step3_complete"  ##< 主灯组与立柱组以 3 Hz 闪烁。
    STEP4_COMPLETE = "step4_complete"  ##< 主灯组与立柱组常亮。
    STEP5_ROTATING = "step5_rotating"  ##< 指示目标角度与当前角度。
    STEP5_COMPLETE = "step5_complete"  ##< 以绿色指示目标角度。
    RECOVERING = "recovering"  ##< 所有分组以 3 Hz 闪烁。


GROUP1 = tuple(f"core_back_light_bar_{i}" for i in range(1, 37))  ##< 角度显示灯条。
GROUP2 = tuple(f"core_back_main_light_{i}" for i in range(1, 3))  ##< 主灯。
GROUP3 = tuple(f"core_pillar_L_light_{i}" for i in range(1, 7)) + tuple(
    f"core_pillar_side_light_{i}" for i in range(1, 4)
)  ##< 立柱灯。


@dataclass(frozen=True)
class LightConfig:
    """@brief 规则未规定的颜色与时序的仿真假设。"""

    duty_cycle: float = 0.5  ##< 闪烁周期内通电的时间占比。
    chase_hz: float = 1.0  ##< 每秒完整扫过的次数。
    chase_width: int = 3  ##< 同时点亮的相邻灯条数。
    red: tuple = (1.0, 0.005, 0.012)  ##< 红方线性 RGB。
    blue: tuple = (0.005, 0.08, 1.0)  ##< 蓝方线性 RGB。
    white: tuple = (1.0, 1.0, 1.0)  ##< 第一组的线性 RGB。
    green: tuple = (0.005, 1.0, 0.02)  ##< 角度指示成功时的线性 RGB。
    off: tuple = (0.015, 0.015, 0.015)  ##< 未通电时的漫反射线性 RGB。


@dataclass(frozen=True)
class Lamp:
    """@brief 赋给单个灯网格的材质。"""

    rgb: tuple  ##< 线性 RGB。
    on: bool  ##< 该灯是否发光。


@cache
def _bar_angles() -> np.ndarray:
    """@brief 测量各灯条中心绕装配旋转轴的角度，核心系，单位 rad。"""
    centres = []
    for index in range(1, 37):
        vertices, _ = read_obj(MODEL_DIR / "meshes" / f"back_light_bar_{index}.obj")
        centres.append(vertices.mean(axis=0))
    centres = np.asarray(centres)
    return np.arctan2(-centres[:, 1], centres[:, 2] - 0.208)


class LightController:
    """@brief 在任意时刻计算规则定义的灯光分组状态。"""

    def __init__(self, config: LightConfig = LightConfig()):
        """@brief 设置规则表仿真参数。

        @param[in] config 时序与材质假设。
        """
        self.config = config

    def angle_bar(self, angle: float) -> str:
        """@brief 把显示角度量化到最近的实体灯条。

        @param[in] angle 绕核心系 +X 的右手旋转角，单位 rad。
        @return 几何名；距离相等时取下标较小者。
        """
        if not -np.pi / 2 <= angle <= np.pi / 2:
            raise ValueError("Indicated rotation must be within [-pi/2, pi/2] rad")
        distances = np.abs(_bar_angles() - angle)
        return GROUP1[np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-9))[0]]

    def evaluate(self, state: LightState, team: Team, time_s: float, target_angle: float = 0, current_angle: float = 0):
        """@brief 计算 47 个灯的状态，不推进仿真时钟。

        @param[in] state 装配灯光状态。
        @param[in] team 指示的队伍颜色。
        @param[in] time_s 灯光绝对时刻，单位 s。
        @param[in] target_angle 绕核心系 +X 的目标旋转角，单位 rad。
        @param[in] current_angle 显示的绕核心系 +X 的当前旋转角，单位 rad。
        @return 按 MJCF 几何名索引的灯材质。
        """
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Lighting time must be finite and nonnegative (s)")
        config = self.config
        color = config.red if team == Team.RED else config.blue
        off = Lamp(config.off, False)
        lamps = dict.fromkeys(GROUP1, off) | dict.fromkeys(GROUP2 + GROUP3, Lamp(color, True))

        def blink(group, rgb, frequency):
            """@brief 在半周期/整周期的精确边界处计算方波。"""
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
