"""生成器集成测试的运行条件。"""

import os
import shutil

import pytest


@pytest.fixture(scope="module")
def require_blender():
    """@brief 找不到配置的 Blender 可执行文件时跳过渲染测试。"""
    if not shutil.which(os.getenv("BLENDER_EXECUTABLE") or "blender"):
        pytest.skip("Blender is absent")
