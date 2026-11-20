"""@brief 原图坐标下的渲染帧与数据集记录。"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .camera import CameraConfig
from .lighting import LightState, Team


@dataclass
class Frame:
    """@brief 单个核心的图像、光学深度及对齐的真值，均为独立持有。"""

    rgb: np.ndarray  ##< RGB8 图像。
    depth: np.ndarray  ##< 光学深度，单位 m，背景为零。
    camera: CameraConfig  ##< 光学标定。
    camera_from_core: np.ndarray  ##< 核心系在相机光学系中的位姿，平移单位 m。
    pixels: np.ndarray  ##< 有序的 16×2 原图坐标，单位 pixel，相机后方为 NaN。
    visibility: np.ndarray  ##< 0 为图像外，1 为被遮挡，2 为无遮挡。
    renderer: str  ##< 生成该帧的渲染器。
    light_state: LightState  ##< 显示的装配状态。
    team: Team  ##< 指示的队伍。
    time: float  ##< 灯光绝对时刻，单位 s。
    appearance: dict | None  ##< 记录的 Blender 外观覆盖参数；为 None 时使用影棚默认值。

    @property
    def mask(self) -> np.ndarray:
        """@brief 由光学深度提取目标轮廓。

        @return 原图大小的布尔掩码。
        """
        return self.depth > 0

    @property
    def bbox_xyxy(self) -> list[float] | None:
        """@brief 计算轮廓与图像内标注的包围框。

        @return 裁剪到图像内的 XYXY 包围框（单位 pixel，上界不含）；视野内无目标时为 None。
        """
        y, x = np.nonzero(self.mask)
        points = self.pixels[self.visibility > 0]
        x, y = np.concatenate((x, points[:, 0])), np.concatenate((y, points[:, 1]))
        if not len(x):
            return None
        return [
            float(x.min()),
            float(y.min()),
            float(min(self.camera.width, x.max() + 1)),
            float(min(self.camera.height, y.max() + 1)),
        ]

    def to_record(self, identity: str, split: str, source_view: str) -> dict:
        """@brief 按 data_generator Readme 的 schema 序列化一帧原始数据。

        @param[in] identity 唯一记录 ID，同时作为 PNG 文件名主干。
        @param[in] split 源视角所属的数据集划分。
        @param[in] source_view 源视角 ID，之后的增强样本共用该 ID。
        @return 路径相对于数据集根目录、相机后方点为 null 的清单记录。
        """
        return {
            "id": identity,
            "split": split,
            "source_view": source_view,
            "renderer": self.renderer,
            "image": f"images/{identity}.png",
            "mask": f"masks/{identity}.png",
            "width": self.camera.width,
            "height": self.camera.height,
            "camera_matrix": self.camera.matrix.tolist(),
            "distortion": [0.0] * 5,
            "camera_from_core": self.camera_from_core.tolist(),
            "pixels": [point.tolist() if np.isfinite(point).all() else None for point in self.pixels],
            "visibility": self.visibility.tolist(),
            "bbox_xyxy": self.bbox_xyxy,
            "light_state": self.light_state.value,
            "team": self.team.value,
            "time": self.time,
            "appearance": self.appearance,
            "augmentation": None,
            "source": None,
            "negative": False,
        }

    def overlay(self) -> Image.Image:
        """@brief 在可见点上画带编号的圆环，在被遮挡点上画叉。

        @return 标注后的 RGB 预览图。
        """
        image = Image.fromarray(self.rgb).copy()
        draw = ImageDraw.Draw(image)
        for index, ((x, y), flag) in enumerate(zip(self.pixels, self.visibility)):
            if flag == 0:
                continue
            if flag == 2:
                color = "lime"
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color)
            else:
                color = "magenta"
                draw.line((x - 3, y - 3, x + 3, y + 3), fill=color)
                draw.line((x - 3, y + 3, x + 3, y - 3), fill=color)
            draw.text((x + 4, y - 8), str(index), fill=color, stroke_width=1, stroke_fill="black")
        return image
