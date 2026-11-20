"""规则表 5-13 的颜色分组与时序边界。"""

import numpy as np
import pytest
from tech_core_data_generator import LightState, Team
from tech_core_data_generator.lighting import GROUP1, GROUP2, GROUP3, LightController


@pytest.mark.parametrize("team", list(Team))
def test_all_rule_states(team):
    """@brief 在 0.2 s 时逐行比对规则表与预期的灯数量。"""
    controller = LightController()
    color = (1, 0.005, 0.012) if team == Team.RED else (0.005, 0.08, 1)
    # 0.2 s 时 1 Hz 分组为亮，3 Hz 分组为灭。
    counts = [(0, 2, 9), (3, 2, 9), (36, 2, 9), (36, 2, 0), (36, 0, 0), (36, 2, 9), (2, 2, 9), (1, 2, 9), (0, 0, 0)]
    for state, expected in zip(LightState, counts, strict=True):
        lamps = controller.evaluate(state, team, 0.2, target_angle=-np.pi / 2, current_angle=np.pi / 2)
        assert tuple(sum(lamps[n].on for n in group) for group in (GROUP1, GROUP2, GROUP3)) == expected
        for name in GROUP2 + GROUP3:
            assert lamps[name].rgb == (color if lamps[name].on else controller.config.off)
        if state == LightState.STEP5_ROTATING:
            assert lamps[GROUP1[0]].rgb == (1, 1, 1) and lamps[GROUP1[-1]].rgb == color
        if state == LightState.STEP5_COMPLETE:
            assert lamps[GROUP1[-1]].rgb == controller.config.green


def test_blink_edges_and_chase_wrap():
    """@brief 测试半周期与扫描结束边界处的绝对时刻。"""
    controller = LightController()
    for hz, state in [(1, LightState.READY), (3, LightState.RECOVERING)]:
        for time, on in [(1 / hz, True), (0.5 / hz, False), (0, True)]:
            lamps = controller.evaluate(state, Team.RED, time)
            assert all(lamp.on == on for lamp in lamps.values())
    lamps = controller.evaluate(LightState.MOVING, Team.BLUE, 35 / 36)
    assert {name for name in GROUP1 if lamps[name].on} == {GROUP1[-1], GROUP1[0], GROUP1[1]}


def test_invalid_time_and_rotation():
    """@brief 拒绝超出规则范围的时刻与显示角度。"""
    controller = LightController()
    with pytest.raises(ValueError, match="time"):
        controller.evaluate(LightState.READY, Team.RED, -0.1)
    with pytest.raises(ValueError, match="rotation"):
        controller.evaluate(LightState.STEP5_COMPLETE, Team.RED, 0, target_angle=2.0)
