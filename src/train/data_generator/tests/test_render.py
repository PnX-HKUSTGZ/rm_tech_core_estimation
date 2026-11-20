"""基于真实离屏渲染器的集成测试。"""

import os
import shutil

import numpy as np
import pytest
from PIL import Image
from tech_core_data_generator import CameraConfig, TechCoreSimulator, Viewpoint
from tech_core_data_generator.__main__ import main
from tech_core_data_generator.appearance import sample_appearance


@pytest.fixture(scope="module")
def mujoco_sim():
    """@brief 深度与标注检查共用同一个 EGL 上下文。"""
    with TechCoreSimulator() as simulator:
        yield simulator


@pytest.fixture(scope="module")
def blender_sim():
    """@brief 多帧共用一个 Cycles 场景，测试其持久状态。"""
    if not os.getenv("BLENDER_EXECUTABLE") and not shutil.which("blender"):
        pytest.skip("Blender is absent")
    with TechCoreSimulator("blender", blender_samples=2) as simulator:
        yield simulator


def test_visibility_clipping_and_behind_camera(mujoco_sim):
    """@brief 区分缺失的图像坐标与有效但被遮挡的标注。"""
    assert np.all(mujoco_sim.render().visibility == 2)
    frame = mujoco_sim.render(Viewpoint(0.2, 0.9, -0.6))
    assert frame.visibility[[0, 5, 9]].tolist() == [2, 1, 0]
    assert frame.bbox_xyxy[2:] == [CameraConfig.width, CameraConfig.height]
    behind = mujoco_sim.render(Viewpoint(target=(2, 0, 0)))
    record = behind.to_record("behind", "stress", "behind")
    assert record["pixels"] == [None] * 16
    assert record["visibility"] == [0] * 16
    assert record["bbox_xyxy"] is None and not behind.mask.any()
    with pytest.raises(ValueError, match="Blender"):
        mujoco_sim.render(appearance={"exposure": 0})


@pytest.mark.parametrize("view", [Viewpoint(), Viewpoint(0.65, 0.5, 0.3, roll=0.15)])
def test_matching_optical_depth_and_silhouette(mujoco_sim, blender_sim, view):
    """@brief 比较远离栅格边缘处的深度及轮廓交并比。"""
    a, b = [
        sim.render(view, "step5_rotating", "blue", target_angle=-0.6, current_angle=0.6)
        for sim in (mujoco_sim, blender_sim)
    ]
    np.testing.assert_array_equal(a.pixels, b.pixels)
    np.testing.assert_array_equal(a.visibility, b.visibility)
    overlap = a.mask & b.mask
    assert overlap.sum() / (a.mask | b.mask).sum() > 0.97
    interior = overlap.copy()
    for axis in (0, 1):
        for shift in (-1, 1):
            interior &= np.roll(overlap, shift, axis) & (abs(a.depth - np.roll(a.depth, shift, axis)) < 0.01)
    assert np.quantile(abs(a.depth[interior] - b.depth[interior]), 0.95) < 0.004
    assert a.rgb.std() > 10 and b.rgb.std() > 10


def test_blender_appearance_resets_between_frames(blender_sim):
    """@brief 在同一工作进程上依次运行影棚、随机化、影棚三种外观。"""
    view = Viewpoint(0.8, 0.2, 0.15)
    first = blender_sim.render(view)
    changed = blender_sim.render(view, appearance=sample_appearance(np.random.default_rng(3), view))
    assert np.mean(np.abs(first.rgb.astype(float) - changed.rgb)) > 1
    np.testing.assert_array_equal(first.visibility, changed.visibility)
    np.testing.assert_allclose(first.depth, changed.depth, atol=1e-6)
    np.testing.assert_array_equal(first.rgb, blender_sim.render(view).rgb)


def test_preview_writes_original_and_overlay(tmp_path):
    """@brief 校验预览入口输出的两种图像产物。"""
    main(["preview", "--out", str(tmp_path), "--width", "96", "--height", "72", "--roll", "0.1"])
    assert {path.name for path in tmp_path.iterdir()} == {"frame.png", "overlay.png"}
    with Image.open(tmp_path / "frame.png") as rgb, Image.open(tmp_path / "overlay.png") as overlay:
        assert rgb.size == overlay.size == (96, 72)
        assert np.any(np.asarray(rgb) != np.asarray(overlay))
