"""@brief Persistent headless Blender process for Cycles RGB and optical depth."""

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
    """@brief Locate the headless Blender executable.

    @return Executable selected by BLENDER_EXECUTABLE, or blender on PATH.

    @throws FileNotFoundError When the selected executable cannot be run.
    """
    command = os.getenv("BLENDER_EXECUTABLE") or "blender"
    binary = shutil.which(command)
    if binary is None:
        raise FileNotFoundError(f"Cannot find {command!r}. Set BLENDER_EXECUTABLE or put Blender on PATH.")
    return binary


class BlenderBackend:
    """@brief Reuse one Blender process and its loaded scene for a sequence of frames."""

    def __init__(self, camera, samples):
        """@brief Start the worker and load the static studio scene.

        @param[in] camera Optical calibration (pixel) and clipping distances (m).
        @param[in] samples Cycles samples per pixel.
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
        """@brief Close the request pipe and reap the worker, including during a cancelled render."""
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def _receive(self):
        """@brief Read one worker reply, retaining Blender's diagnostic output if it exits."""
        log = deque(maxlen=30)
        for line in self.process.stdout:
            if line.startswith("TECH_CORE_RESULT "):
                return json.loads(line.removeprefix("TECH_CORE_RESULT "))
            log.append(line.rstrip())
        raise RuntimeError("Blender worker exited before replying.\n" + "\n".join(log))

    def render(self, camera_from_core, lamps, appearance):
        """@brief Render RGB8 and optical depth (m, background zero) in Cycles.

        @param[in] camera_from_core Core pose in the optical frame (translation in m).
        @param[in] lamps Materials indexed by mesh name.
        @param[in] appearance Blender overrides or None for the default studio.
        @return Owned RGB and optical depth arrays.
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
        """@brief Close the worker and remove its private transfer directory."""
        self.resources.close()
