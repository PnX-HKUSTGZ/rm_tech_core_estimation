"""@brief 常驻的无界面 Blender 进程，用 Cycles 渲染 RGB 与光学深度。"""

import json
import os
import shutil
import subprocess
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from ..appearance import DEFAULT_APPEARANCE
from ..geometry import MODEL_DIR


def find_blender() -> str:
    """@brief 定位无界面 Blender 可执行文件。

    @return BLENDER_EXECUTABLE 指定的可执行文件，否则为 PATH 中的 blender。

    @throws FileNotFoundError 所选可执行文件无法运行时抛出。
    """
    command = os.getenv("BLENDER_EXECUTABLE") or "blender"
    binary = shutil.which(command)
    if binary is None:
        raise FileNotFoundError(f"Cannot find {command!r}. Set BLENDER_EXECUTABLE or put Blender on PATH.")
    return binary


class BlenderBackend:
    """@brief 复用同一个 Blender 进程及其已加载场景渲染一系列帧。"""

    def __init__(self, camera, samples):
        """@brief 启动工作进程并加载静态影棚场景。

        @param[in] camera 光学标定（单位 pixel）与裁剪距离（单位 m）。
        @param[in] samples Cycles 每像素采样数。
        """
        binary = find_blender()
        with ExitStack() as resources:
            self.work = Path(resources.enter_context(TemporaryDirectory(prefix="tech-core-blender-")))
            config_path = self.work / "config.json"
            config_path.write_text(
                json.dumps({"camera": asdict(camera), "samples": samples, "model_dir": str(MODEL_DIR)})
            )
            worker = Path(__file__).with_name("blender_worker.py")
            self.process = resources.enter_context(
                subprocess.Popen(
                    [
                        binary,
                        "--background",
                        "--factory-startup",
                        "--threads",
                        "4",
                        "--python-exit-code",
                        "1",
                        "--python",
                        str(worker),
                        "--",
                        str(config_path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            )
            resources.callback(self._stop)
            self.version = self._receive()["version"]
            self.resources = resources.pop_all()

    def _stop(self):
        """@brief 关闭请求管道并回收工作进程，渲染被取消时同样适用。"""
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def _receive(self):
        """@brief 读取一次工作进程回复；进程退出时保留 Blender 的诊断输出。"""
        log = deque(maxlen=30)
        for line in self.process.stdout:
            if line.startswith("TECH_CORE_RESULT "):
                return json.loads(line.removeprefix("TECH_CORE_RESULT "))
            log.append(line.rstrip())
        raise RuntimeError("Blender worker exited before replying.\n" + "\n".join(log))

    def render(self, camera_from_core, lamps, appearance):
        """@brief 用 Cycles 渲染 RGB8 与光学深度（单位 m，背景为零）。

        @param[in] camera_from_core 核心系在相机光学系中的位姿，平移单位 m。
        @param[in] lamps 按网格名索引的材质。
        @param[in] appearance Blender 外观覆盖参数；为 None 时使用默认影棚。
        @return 独立持有的 RGB 与光学深度数组。
        """
        overrides = appearance or {}
        if unknown := overrides.keys() - DEFAULT_APPEARANCE.keys():
            raise ValueError(f"Unknown Blender appearance parameters: {sorted(unknown)}")
        request = {
            "camera_from_core": camera_from_core.tolist(),
            "lamps": {name: asdict(lamp) for name, lamp in lamps.items()},
            "appearance": DEFAULT_APPEARANCE | overrides,
        }
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        self._receive()
        with Image.open(self.work / "rgb.png") as image:
            rgb = np.array(image.convert("RGB"))
        return rgb, np.load(self.work / "depth.npy", allow_pickle=False)

    def close(self):
        """@brief 关闭工作进程并删除其私有传输目录。"""
        self.resources.close()
