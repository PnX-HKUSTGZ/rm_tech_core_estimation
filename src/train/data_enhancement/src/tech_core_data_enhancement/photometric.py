"""@brief Colour, lighting and camera-response augmentation in image coordinates."""

import cv2
import numpy as np


def _photometric(rgb: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """@brief Apply the shared appearance policy and return the sampled operation parameters."""
    if rng.random() < 0.15:
        return rgb.copy(), {}

    image = rgb.astype(np.float32) / 255
    maximum, minimum = image.max(axis=2), image.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 0.01)
    # Keep bright coloured lamps and white highlights distinguishable as exposure falls.
    emission = np.clip((maximum - 0.55) / 0.35, 0, 1) * np.maximum(saturation, np.clip((minimum - 0.8) / 0.2, 0, 1))
    gamma = rng.uniform(0.75, 2.6)
    gain = float(np.exp(rng.uniform(np.log(0.32), np.log(1.5))))
    lamp_fraction = rng.uniform(0.1, 0.85)
    preserve = emission[:, :, None] * lamp_fraction
    image = image**gamma * gain * (1 - preserve) + image * preserve
    params = {"gamma": gamma, "gain": gain, "lamp_fraction": lamp_fraction}

    if rng.random() < 0.5:
        balance = rng.uniform(0.8, 1.2, 3)
        image *= balance
        params["white_balance_rgb"] = balance.tolist()
    if rng.random() < 0.15:
        image = 0.2 * image + 0.8 * image.mean(axis=2, keepdims=True)
        params["grayscale_fraction"] = 0.8
    if rng.random() < 0.35:
        height, width = rgb.shape[:2]
        dx, dy = rng.uniform(-0.4, 0.4, 2).tolist()
        field = 1 + dx * np.linspace(-1, 1, width) + dy * np.linspace(-1, 1, height)[:, None]
        image *= np.clip(field, 0.35, 1.6)[:, :, None]
        params["illumination_gradient_xy"] = [dx, dy]

    params["bias"] = rng.uniform(-8, 8)
    image = image * 255 + params["bias"]
    if rng.random() < 0.25:
        params["noise_std"] = rng.uniform(0.2, 2.0)
        image += rng.normal(0, params["noise_std"], image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        hue_steps = int(rng.integers(-4, 5))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_steps) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        params["hue_rad"] = hue_steps * np.pi / 90
    if rng.random() < 0.08:
        params["blur_sigma_px"] = 0.4
        image = cv2.GaussianBlur(image, (3, 3), params["blur_sigma_px"])
    if rng.random() < 0.12:
        params["jpeg_quality"] = int(rng.integers(75, 101))
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg_quality"]])
        image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image, params


def photometric(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """@brief Apply training-time appearance augmentation at fixed image coordinates.

    @param[in] rgb H×W×3 RGB uint8 image.
    @param[in,out] rng NumPy generator for this training sample.
    @return New RGB uint8 image with the same shape and geometry.
    """
    return _photometric(rgb, rng)[0]
