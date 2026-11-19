"""@brief Rendered frame and dataset record in the original image coordinates."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .camera import CameraConfig
from .lighting import LightState, Team


@dataclass
class Frame:
    """@brief Owned image, optical depth and aligned ground truth for a single core."""

    rgb: np.ndarray  ##< RGB8 image.
    depth: np.ndarray  ##< Optical depth (m) with zero background.
    camera: CameraConfig  ##< Optical calibration.
    camera_from_core: np.ndarray  ##< Core pose in the optical frame (translation in m).
    pixels: np.ndarray  ##< Ordered 16×2 original image coordinates (pixel), NaN behind camera.
    visibility: np.ndarray  ##< 0 outside, 1 occluded, 2 unobstructed.
    renderer: str  ##< Renderer that produced this frame.
    light_state: LightState  ##< Displayed assembly state.
    team: Team  ##< Indicated team.
    time: float  ##< Absolute lamp time (s).
    appearance: dict | None  ##< Recorded Blender overrides, None for studio defaults.

    @property
    def mask(self) -> np.ndarray:
        """@brief Extract the target silhouette from optical depth.

        @return Boolean mask in the original image.
        """
        return self.depth > 0

    @property
    def bbox_xyxy(self) -> list[float] | None:
        """@brief Bound the silhouette and in-frame labels.

        @return Clipped XYXY bounds (pixel, exclusive upper edges), or None for an empty view.
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
        """@brief Serialize one raw frame using the data_generator Readme schema.

        @param[in] identity Unique record ID used as the PNG stem.
        @param[in] split Dataset partition of the source view.
        @param[in] source_view Source view ID, shared by any later augmentations.
        @return Manifest record with dataset-relative paths and null behind-camera points.
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
        """@brief Draw numbered rings on visible points and crosses on occluded points.

        @return Annotated RGB preview.
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
